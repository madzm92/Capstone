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
COMMUTER_RAIL_SPLITS = [0, 50, 100]  # percent using commuter rail

parcels = gpd.read_postgis(
    f"""
    SELECT s.town_name, s.acres as area_acres, s.loc_id as PID, s.geometry, 
           s.use_type, s.transit, s.sqft, s.total_excluded, tn.min_multi_family
    FROM general_data.shapefiles s
    LEFT JOIN general_data.town_nameplate tn ON s.town_name = tn.town_name
    WHERE s.town_name = '{TOWN}' AND (s.total_excluded IS NULL OR s.total_excluded = 0)
    """,
    engine,
    geom_col="geometry"
)

if parcels.crs is None:
    parcels.set_crs(epsg=26986, inplace=True)

parcels['centroid'] = parcels.geometry.centroid

traffic = gpd.read_postgis(
    f"""
    SELECT tn.geom as geometry, tn.location_id as sensor_id, tn.functional_class
    FROM general_data.traffic_nameplate tn 
    WHERE tn.town_name = '{TOWN}'
    """,
    engine,
    geom_col="geometry"
)

commuter_rail = gpd.read_postgis(
    f"""
    SELECT stop_name, town_name, geometry
    FROM general_data.commuter_rail_stops
    """,
    engine,
    geom_col="geometry"
)

traffic_hist = pd.read_sql(f"""
    SELECT tn.town_name, tn.location_id as sensor_id, 
           tc.start_date_time as timestamp, tc.hourly_count as volume
    FROM general_data.traffic_nameplate tn  
    LEFT JOIN general_data.traffic_counts tc ON tn.location_id = tc.location_id
    WHERE tn.town_name = '{TOWN}'
""", engine)

pop_hist = pd.read_sql(f"""
    SELECT ap.year, tcc.town_name, ap.total_population as population
    FROM general_data.annual_population ap
    LEFT JOIN general_data.town_census_crosswalk tcc ON ap.zip_code = tcc.zip_code
    WHERE tcc.town_name = '{TOWN}'
""", engine)

print("Preprocessing traffic data by weekday...")

traffic_hist['datetime'] = pd.to_datetime(traffic_hist['timestamp'])
traffic_hist['weekday'] = traffic_hist['datetime'].dt.dayofweek
traffic_hist['date'] = traffic_hist['datetime'].dt.date
traffic_hist = traffic_hist.dropna(subset=['date'])

daily_traffic = (
    traffic_hist.groupby(['date', 'sensor_id', 'weekday'])['volume']
    .sum()
    .reset_index()
)

daily_traffic['year'] = pd.to_datetime(daily_traffic['date']).dt.year
pop_hist['year'] = pop_hist['year'].astype(int)
daily_traffic = daily_traffic.merge(pop_hist, on='year', how='left')
daily_traffic = daily_traffic.merge(traffic[['sensor_id', 'functional_class']], on='sensor_id', how='left')

print("Training Random Forest models per sensor...")
sensor_models = {}
for sensor_id, group in daily_traffic.groupby('sensor_id'):
    if len(group) < MIN_DATA_POINTS:
        continue

    sensor_geom = traffic.loc[traffic['sensor_id'] == sensor_id, 'geometry'].iloc[0]
    distances_km = parcels['centroid'].distance(sensor_geom) / 1000
    avg_distance_km = distances_km.mean()

    X = group[['population', 'weekday']].copy()
    X['functional_class'] = group['functional_class'].astype('category').cat.codes
    X['distance_km'] = avg_distance_km
    y = group['volume'].values

    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)
    sensor_models[sensor_id] = model

model_inputs = []
model_outputs = []
baseline_pop = pop_hist['population'].iloc[-1]

print("Running traffic scenarios...")
for _, parcel in parcels.iterrows():
    parcel_id = parcel['pid']
    min_multi_family = parcel.get('min_multi_family', 0) or 0
    centroid = parcel['centroid']
    distances_to_sensors = traffic.geometry.distance(centroid) / 1000
    distances_to_rail = commuter_rail.geometry.distance(centroid)
    nearest_rail_stop = commuter_rail.iloc[distances_to_rail.idxmin()]

    for pct in POP_PCT_OPTIONS:
        new_units = min_multi_family * (pct / 100)
        new_people = new_units * AVG_HH_SIZE
        new_trips = new_people * TRIPS_PER_PERSON

        for rail_pct in COMMUTER_RAIL_SPLITS:
            highway_pct = 100 - rail_pct
            scenario_id = str(uuid.uuid4())[:8] + f"_{rail_pct}"
            rail_trips = new_trips * (rail_pct / 100)
            highway_trips = new_trips * (highway_pct / 100)

            model_inputs.append({
                "scenario_id": scenario_id,
                "parcel_id": parcel_id,
                "pop_increase_pct": pct,
                "new_units": round(new_units, 2),
                "new_people": round(new_people, 1),
                "new_trips": round(new_trips, 1),
                "rail_pct": rail_pct,
                "rail_stop": nearest_rail_stop['stop_name'],
                "use_type": parcel['use_type'],
                "transit": parcel['transit'],
                "sqft": parcel['sqft'],
                "total_excluded": parcel['total_excluded']
            })

            weights = 1 / np.power(distances_to_sensors + 1e-6, ALPHA)
            weights /= weights.sum()
            weekday = 2  # Tuesday

            for i, row in traffic.iterrows():
                sensor_id = row['sensor_id']
                func_class_code = pd.Series([row['functional_class']]).astype('category').cat.codes[0]
                model = sensor_models.get(sensor_id)
                added_trips = highway_trips * weights.iloc[i]

                if model:
                    pred_input = [[
                        baseline_pop,
                        weekday,
                        func_class_code,
                        distances_to_sensors.iloc[i]
                    ]]
                    baseline_pred = model.predict(pred_input)[0]
                else:
                    baseline_pred = daily_traffic['volume'].mean()

                pct_increase = (added_trips / baseline_pred) * 100 if baseline_pred > 0 else 0

                model_outputs.append({
                    "scenario_id": scenario_id,
                    "sensor_id": sensor_id,
                    "model": "RandomForest",
                    "weekday": weekday,
                    "added_trips": round(added_trips, 2),
                    "pct_increase": round(pct_increase, 2),
                    "distance_km": round(distances_to_sensors.iloc[i], 3),
                    "functional_class": row['functional_class']
                })

print(f"Processed {len(model_inputs)} scenarios across {len(parcels)} parcels")

model_inputs_df = pd.DataFrame(model_inputs)
model_outputs_df = pd.DataFrame(model_outputs)

model_inputs_df.to_csv("model_inputs_boxford_rf.csv", index=False)
model_outputs_df.to_csv("model_outputs_boxford_rf.csv", index=False)
print("Random Forest weekday-level model results with functional class and land classification saved.")
