# traffic_model_local_roads_all_towns_with_deltas.py
# Predict traffic on Local Roads across all towns using annual population with shared model
# Includes modeling of both absolute avg daily volume and year-to-year traffic volume changes (deltas)

import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# DB connection
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')

MIN_DATA_POINTS = 3
COVID_YEARS = [2020, 2021]
FUNCTIONAL_CLASS_FILTER = '(7) Local Road or Street'
MAX_VOLUME = 2_000_000

print("Loading traffic sensors for all towns...")
traffic = gpd.read_postgis(
    f"""
    SELECT location_id as sensor_id, town_name, functional_class, geom
    FROM general_data.traffic_nameplate
    WHERE functional_class = '{FUNCTIONAL_CLASS_FILTER}'
    """,
    engine,
    geom_col="geom"
)

print("Loading historical traffic counts...")
traffic_hist = pd.read_sql(f"""
    SELECT tn.location_id as sensor_id, tc.start_date_time as timestamp, tc.hourly_count as volume
    FROM general_data.traffic_nameplate tn  
    LEFT JOIN general_data.traffic_counts tc ON tn.location_id = tc.location_id
    WHERE tn.functional_class = '{FUNCTIONAL_CLASS_FILTER}'
""", engine)

traffic_hist['datetime'] = pd.to_datetime(traffic_hist['timestamp'])
traffic_hist = traffic_hist.dropna(subset=['datetime', 'volume'])
traffic_hist['weekday'] = traffic_hist['datetime'].dt.dayofweek
traffic_hist['year'] = traffic_hist['datetime'].dt.year
traffic_hist['date'] = traffic_hist['datetime'].dt.date
traffic_hist['is_weekend'] = (traffic_hist['weekday'] >= 5).astype(int)
traffic_hist['is_covid_year'] = traffic_hist['year'].isin(COVID_YEARS).astype(int)

print("Aggregating daily traffic and computing annual average daily traffic per sensor...")
daily_traffic = (
    traffic_hist.groupby(['sensor_id', 'year', 'date'])['volume']
    .sum()
    .reset_index(name='daily_volume')
)

avg_daily_traffic = (
    daily_traffic.groupby(['sensor_id', 'year'])['daily_volume']
    .mean()
    .reset_index(name='avg_daily_volume')
)

avg_daily_traffic = avg_daily_traffic.merge(traffic[['sensor_id', 'town_name', 'functional_class']], on='sensor_id', how='left')
avg_daily_traffic = avg_daily_traffic[avg_daily_traffic['avg_daily_volume'] < MAX_VOLUME]

print("Loading population history for all towns...")
pop_hist = pd.read_sql(f"""
    SELECT ap.year, tcc.town_name, ap.total_population as population
    FROM general_data.annual_population ap
    LEFT JOIN general_data.town_census_crosswalk tcc ON ap.zip_code = tcc.zip_code
""", engine)
pop_hist['year'] = pop_hist['year'].astype(int)

avg_daily_traffic = avg_daily_traffic.merge(pop_hist, on=['town_name', 'year'], how='left')
avg_daily_traffic = avg_daily_traffic.dropna(subset=['population'])

sensor_counts = avg_daily_traffic['sensor_id'].value_counts()
valid_sensors = sensor_counts[sensor_counts >= MIN_DATA_POINTS].index
avg_daily_traffic = avg_daily_traffic[avg_daily_traffic['sensor_id'].isin(valid_sensors)]

print("Encoding categorical features...")
le_sensor = LabelEncoder()
le_town = LabelEncoder()
le_class = LabelEncoder()

avg_daily_traffic['sensor_code'] = le_sensor.fit_transform(avg_daily_traffic['sensor_id'])
avg_daily_traffic['town_code'] = le_town.fit_transform(avg_daily_traffic['town_name'])
avg_daily_traffic['class_code'] = le_class.fit_transform(avg_daily_traffic['functional_class'])

# Features for absolute volume model
features_abs = ['population', 'year', 'sensor_code', 'town_code', 'class_code']
X_abs = avg_daily_traffic[features_abs]
y_abs = avg_daily_traffic['avg_daily_volume']

scaler_abs = StandardScaler()
X_abs_scaled = scaler_abs.fit_transform(X_abs)

print(f"Training absolute traffic volume model on {len(avg_daily_traffic)} records...")
model_abs = XGBRegressor(n_estimators=100, random_state=42)
model_abs.fit(X_abs_scaled, y_abs)

# --- Predict 5% population increase effect for each sensor's latest year ---
print("Predicting 5% population increase effect for each sensor...")
latest_records = avg_daily_traffic.loc[avg_daily_traffic.groupby('sensor_id')['year'].idxmax()].copy()

# Don't re-transform encoded features — just modify population
modified_records = latest_records.copy()
modified_records['population'] *= 1.05

baseline_inputs = scaler_abs.transform(latest_records[features_abs])
modified_inputs = scaler_abs.transform(modified_records[features_abs])

latest_records['predicted_baseline'] = model_abs.predict(baseline_inputs)
latest_records['predicted_increased'] = model_abs.predict(modified_inputs)
latest_records['pct_change'] = ((latest_records['predicted_increased'] - latest_records['predicted_baseline']) / latest_records['predicted_baseline']) * 100

latest_records[['sensor_id', 'town_name', 'year', 'predicted_baseline', 'predicted_increased', 'pct_change']].to_csv("traffic_increase_predictions_5pct_pop.csv", index=False)
print("Saved traffic predictions with 5% population increase to traffic_increase_predictions_5pct_pop.csv")

breakpoint()
