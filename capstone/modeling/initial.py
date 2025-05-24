# traffic_model_boxford.py
# Predict traffic impacts in Boxford from multifamily housing increases and store results

import pandas as pd
import geopandas as gpd
import numpy as np
import uuid
from shapely.geometry import Point
from matplotlib import pyplot as plt
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sklearn.linear_model import LinearRegression

# Set up the SQLAlchemy engine and session
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')
Session = sessionmaker(bind=engine)
session = Session()

# --- Parameters ---
TOWN = "Boxford"
POP_PCT_OPTIONS = [5, 10, 15, 20]  # % increase options
ZONING_DENSITY = 10  # units per acre assumed for multifamily
AVG_HH_SIZE = 2.4
TRIPS_PER_PERSON = 3.5
ALPHA = 2.0  # decay for spatial weighting

# --- Load Data ---
print("Loading data...")

parcels = gpd.read_postgis(
    """SELECT s.town_name as town, s.acres as area_acres, s.loc_id as PID, s.geometry, tn.min_multi_family
    FROM general_data.shapefiles s
    LEFT JOIN general_data.town_nameplate tn
    ON s.town_name = tn.town_name
    WHERE s.town_name = 'Boxford'""",
    engine,
    geom_col="geometry"
)

if parcels.crs is None:
    parcels.set_crs(epsg=26986, inplace=True) 

# Compute centroid now in projected CRS
parcels['centroid'] = parcels.geometry.centroid

traffic = gpd.read_postgis(
    """SELECT tn.geom as geometry, tn.location_id as sensor_id 
    FROM general_data.traffic_nameplate tn 
    WHERE tn.town_name = 'Boxford'""",
    engine,
    geom_col="geometry"
)

# Load historical traffic and population data
traffic_hist = pd.read_sql("""
                           SELECT tn.town_name, tn.location_id as sensor_id, 
                           tc.start_date_time as timestamp, tc.hourly_count as volume
                           FROM general_data.traffic_nameplate tn  
                           LEFT JOIN general_data.traffic_counts tc
                           ON tn.location_id = tc.location_id
                           WHERE tn.town_name = 'Boxford'""", engine)

pop_hist = pd.read_sql("""SELECT ap.year, tcc.town_name, ap.total_population as population
                       FROM general_data.annual_population ap
                       LEFT JOIN general_data.town_census_crosswalk tcc
                       ON ap.zip_code = tcc.zip_code
                       WHERE tcc.town_name = 'Boxford'""", engine)

# --- Build predictive model based on historical data ---
print("Aggregating traffic data to yearly averages...")
traffic_hist['datetime'] = pd.to_datetime(traffic_hist['timestamp'])
traffic_hist['year'] = traffic_hist['datetime'].dt.year
traffic_hist['date'] = traffic_hist['datetime'].dt.date

#drop where timestamp = null
traffic_hist = traffic_hist.dropna(subset='year')

traffic_hist['year'] = traffic_hist['year'].astype(int)
pop_hist['year'] = pop_hist['year'].astype(int)

# Sum volume per sensor per day
daily_traffic = traffic_hist.groupby(['year', 'date', 'sensor_id'])['volume'].sum().reset_index()

# Average daily traffic per sensor per year
avg_daily_by_sensor_year = daily_traffic.groupby(['year', 'sensor_id'])['volume'].mean().reset_index()

# Merge with population data
hist_data = avg_daily_by_sensor_year.merge(pop_hist, on="year")

# --- Train a regression model per sensor ---
print("Training linear regression models per sensor...")
sensor_models = {}
for sensor_id, group in hist_data.groupby('sensor_id'):
    X = group[['population']].values
    y = group[['volume']].values
    if len(group) > 1:  # Require at least two points to train model
        model = LinearRegression().fit(X, y)
        sensor_models[sensor_id] = model

# --- Initialize output tables ---
model_inputs = []
model_outputs = []

# Get latest population baseline for prediction
baseline_pop = pop_hist['population'].iloc[-1]

# --- Loop through parcels and % minimum multifamily housing ---
for _, parcel in parcels.iterrows():
    parcel_id = parcel['PID'] if 'PID' in parcel else parcel.name
    min_multi_family = parcel['min_multi_family'] if 'min_multi_family' in parcel else 0

    for pct in POP_PCT_OPTIONS:
        scenario_id = str(uuid.uuid4())[:8]

        new_units = min_multi_family * (pct / 100)
        new_people = new_units * AVG_HH_SIZE
        new_trips = new_people * TRIPS_PER_PERSON

        model_inputs.append({
            "scenario_id": scenario_id,
            "parcel_id": parcel_id,
            "pop_increase_pct": pct,
            "new_units": round(new_units, 2),
            "new_people": round(new_people, 1),
            "new_trips": round(new_trips, 1)
        })

        centroid = parcel['centroid']
        distances = traffic.geometry.distance(centroid)
        distances_km = distances / 1000
        weights = 1 / np.power(distances_km + 1e-6, ALPHA)
        weights /= weights.sum()

        for i, row in traffic.iterrows():
            sensor_id = row['sensor_id']
            model = sensor_models.get(sensor_id)

            added_trips = new_trips * weights.iloc[i]

            if model:
                baseline_pred = model.predict([[baseline_pop]])[0][0]
            else:
                # fallback: average of all sensors (or set baseline_pred to 0 or some default)
                baseline_pred = hist_data['volume'].mean()

            pct_increase = (added_trips / baseline_pred) * 100 if baseline_pred > 0 else 0

            model_outputs.append({
                "scenario_id": scenario_id,
                "sensor_id": sensor_id,
                "added_trips": round(added_trips, 2),
                "pct_increase": round(pct_increase, 2),
                "distance_km": round(distances_km.iloc[i], 3)
            })

print(f"Processed {len(model_inputs)} scenarios across {len(parcels)} parcels")


# --- Save results ---
model_inputs_df = pd.DataFrame(model_inputs)
model_outputs_df = pd.DataFrame(model_outputs)

model_inputs_df.to_csv("model_inputs_boxford.csv", index=False)
model_outputs_df.to_csv("model_outputs_boxford.csv", index=False)
print("Model results saved to 'outputs/' folder.")
