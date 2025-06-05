# traffic_model_boxford_rf.py
# Predict traffic impacts in Boxford using Random Forest and store results

import pandas as pd
import geopandas as gpd
import numpy as np
import uuid
from shapely.geometry import Point
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.ensemble import RandomForestRegressor

# Set up the SQLAlchemy engine and session
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')
Session = sessionmaker(bind=engine)
session = Session()

TOWN = "Boxford"
POP_PCT_OPTIONS = [5, 10, 15, 20]
ZONING_DENSITY = 10
AVG_HH_SIZE = 2.4
TRIPS_PER_PERSON = 3.5
ALPHA = 2.0
MIN_DATA_POINTS = 3

print("Loading data...")

parcels = gpd.read_postgis(
    """SELECT s.town_name, s.acres as area_acres, s.loc_id as PID, s.geometry, tn.min_multi_family
        FROM general_data.shapefiles s
        LEFT JOIN general_data.town_nameplate tn ON s.town_name = tn.town_name
        WHERE s.town_name = 'Boxford'""",
    engine,
    geom_col="geometry"
)

if parcels.crs is None:
    parcels.set_crs(epsg=26986, inplace=True)

parcels['centroid'] = parcels.geometry.centroid

traffic = gpd.read_postgis(
    """SELECT tn.geom as geometry, tn.location_id as sensor_id 
        FROM general_data.traffic_nameplate tn 
        WHERE tn.town_name = 'Boxford'""",
    engine,
    geom_col="geometry"
)

traffic_hist = pd.read_sql("""
    SELECT tn.town_name, tn.location_id as sensor_id, 
           tc.start_date_time as timestamp, tc.hourly_count as volume
    FROM general_data.traffic_nameplate tn  
    LEFT JOIN general_data.traffic_counts tc ON tn.location_id = tc.location_id
    WHERE tn.town_name = 'Boxford'
""", engine)

pop_hist = pd.read_sql("""
    SELECT ap.year, tcc.town_name, ap.total_population as population
    FROM general_data.annual_population ap
    LEFT JOIN general_data.town_census_crosswalk tcc ON ap.zip_code = tcc.zip_code
    WHERE tcc.town_name = 'Boxford'
""", engine)

print("Aggregating traffic data to yearly averages...")

traffic_hist['datetime'] = pd.to_datetime(traffic_hist['timestamp'])
traffic_hist['weekday'] = traffic_hist['datetime'].dt.dayofweek  # 0=Monday
traffic_hist['date'] = traffic_hist['datetime'].dt.date
traffic_hist = traffic_hist.dropna(subset=['date'])

# Aggregate to daily totals
daily_traffic = (
    traffic_hist.groupby(['date', 'sensor_id', 'weekday'])['volume']
    .sum()
    .reset_index()
)

# Join population by year
daily_traffic['year'] = pd.to_datetime(daily_traffic['date']).dt.year
pop_hist['year'] = pop_hist['year'].astype(np.int32)
daily_traffic = daily_traffic.merge(pop_hist, on='year', how='left')
breakpoint()
print("Training Random Forest models per sensor...")
sensor_models = {}
for sensor_id, group in daily_traffic.groupby('sensor_id'):
    if len(group) < MIN_DATA_POINTS:
        continue
    X = group[['population', 'weekday']].values
    y = group['volume'].values
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    sensor_models[sensor_id] = model

model_inputs = []
model_outputs = []
baseline_pop = pop_hist['population'].iloc[-1]

print("Running traffic scenarios...")
for _, parcel in parcels.iterrows():
    parcel_id = parcel['pid']
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

            weekday = 2  # Wednesday as representative day
            baseline_pred = model.predict([[baseline_pop, weekday]])[0] if model else daily_traffic['volume'].mean()
            pct_increase = (added_trips / baseline_pred) * 100 if baseline_pred > 0 else 0

            model_outputs.append({
                "scenario_id": scenario_id,
                "sensor_id": sensor_id,
                "model": "RandomForest",
                "weekday": weekday,
                "added_trips": round(added_trips, 2),
                "pct_increase": round(pct_increase, 2),
                "distance_km": round(distances_km.iloc[i], 3)
            })

print(f"Processed {len(model_inputs)} scenarios across {len(parcels)} parcels")

model_inputs_df = pd.DataFrame(model_inputs)
model_outputs_df = pd.DataFrame(model_outputs)

breakpoint()
model_inputs_df.to_csv("model_inputs_boxford_rf.csv", index=False)
model_outputs_df.to_csv("model_outputs_boxford_rf.csv", index=False)
print("Random Forest model results saved.")
