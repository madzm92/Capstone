# traffic_model_local_roads_all_towns.py
# Enhanced validation for Local Road traffic prediction across all towns using XGBoost

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
traffic_hist['is_weekend'] = (traffic_hist['weekday'] >= 5).astype(int)
traffic_hist['is_covid_year'] = traffic_hist['year'].isin(COVID_YEARS).astype(int)

print("Aggregating annual traffic per sensor and town...")
annual_traffic = (
    traffic_hist.groupby(['sensor_id', 'year'])['volume']
    .sum()
    .reset_index()
)

annual_traffic = annual_traffic.merge(traffic[['sensor_id', 'town_name', 'functional_class']], on='sensor_id', how='left')
annual_traffic = annual_traffic[annual_traffic['volume'] < MAX_VOLUME]

print("Loading population history for all towns...")
pop_hist = pd.read_sql(f"""
    SELECT ap.year, tcc.town_name, ap.total_population as population
    FROM general_data.annual_population ap
    LEFT JOIN general_data.town_census_crosswalk tcc ON ap.zip_code = tcc.zip_code
""", engine)
pop_hist['year'] = pop_hist['year'].astype(int)

annual_traffic = annual_traffic.merge(pop_hist, on=['town_name', 'year'], how='left')
annual_traffic = annual_traffic.dropna(subset=['population'])

sensor_counts = annual_traffic['sensor_id'].value_counts()
valid_sensors = sensor_counts[sensor_counts >= MIN_DATA_POINTS].index
annual_traffic = annual_traffic[annual_traffic['sensor_id'].isin(valid_sensors)]

print("Encoding categorical features...")
le_sensor = LabelEncoder()
le_town = LabelEncoder()
le_class = LabelEncoder()

annual_traffic['sensor_code'] = le_sensor.fit_transform(annual_traffic['sensor_id'])
annual_traffic['town_code'] = le_town.fit_transform(annual_traffic['town_name'])
annual_traffic['class_code'] = le_class.fit_transform(annual_traffic['functional_class'])

print(f"Training shared XGBoost model on {len(annual_traffic)} records with encoded features...")

features = ['population', 'year', 'sensor_code', 'town_code', 'class_code']
X = annual_traffic[features]
y = annual_traffic['volume']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

print("Validating model per sensor...")
results = []
residuals_all = []

for sensor_id, group in annual_traffic.groupby('sensor_id'):
    X_val_raw = group[features]
    y_true = group['volume']
    X_val_scaled = scaler.transform(X_val_raw)
    y_pred = model.predict(X_val_scaled)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)

    # Residuals
    residuals_all.extend(y_true - y_pred)

    # Predict 5% pop increase from latest year
    latest_row = group[group['year'] == group['year'].max()].iloc[0]
    modified_input = latest_row[features].copy()
    modified_input['population'] *= 1.05
    baseline_input = scaler.transform([latest_row[features]])
    increased_input = scaler.transform([modified_input])

    baseline_pred = model.predict(baseline_input)[0]
    increased_pred = model.predict(increased_input)[0]
    pct_change = ((increased_pred - baseline_pred) / baseline_pred) * 100 if baseline_pred > 0 else 0

    results.append({
        'sensor_id': sensor_id,
        'town_name': group['town_name'].iloc[0],
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'n_obs': len(group),
        'pct_change_5pct_pop': round(pct_change, 2),
        'baseline_volume': round(baseline_pred, 1),
        'increased_volume': round(increased_pred, 1)
    })

results_df = pd.DataFrame(results)
results_df.to_csv("local_roads_model_results.csv", index=False)
print("\nSaved results to local_roads_model_results.csv")

# Residual analysis
sns.histplot(residuals_all, bins=30, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Prediction Error")
plt.savefig("residual_distribution.png")
plt.clf()

# Sensor-level trend comparison
sensor_trends = []

for sensor_id, group in annual_traffic.groupby('sensor_id'):
    if group['year'].nunique() < 2:
        continue
    sorted_group = group.sort_values('year')
    first = sorted_group.iloc[0]
    last = sorted_group.iloc[-1]
    X_first = scaler.transform([first[features]])
    X_last = scaler.transform([last[features]])
    pred_delta = model.predict(X_last)[0] - model.predict(X_first)[0]
    actual_delta = last['volume'] - first['volume']
    sensor_trends.append({
        'sensor_id': sensor_id,
        'town': first['town_name'],
        'actual_change': round(actual_delta),
        'predicted_change': round(pred_delta),
        'error': round(pred_delta - actual_delta)
    })

trend_df = pd.DataFrame(sensor_trends)
trend_df.to_csv("sensor_trend_comparison.csv", index=False)
print("Saved sensor trend comparison to sensor_trend_comparison.csv")
breakpoint()