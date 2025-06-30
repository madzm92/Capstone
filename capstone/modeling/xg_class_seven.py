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

# --- Prepare delta data (year-to-year changes) ---
print("Preparing delta dataset for year-to-year traffic and population changes...")

delta_rows = []
for sensor_id, group in avg_daily_traffic.groupby('sensor_id'):
    group_sorted = group.sort_values('year').reset_index(drop=True)
    for i in range(1, len(group_sorted)):
        prev = group_sorted.loc[i - 1]
        curr = group_sorted.loc[i]

        year_diff = curr['year'] - prev['year']
        pop_change = curr['population'] - prev['population']
        traffic_change = curr['avg_daily_volume'] - prev['avg_daily_volume']

        # Only keep deltas with positive year_diff and reasonable volume
        if year_diff > 0 and curr['avg_daily_volume'] < MAX_VOLUME and prev['avg_daily_volume'] < MAX_VOLUME:
            delta_rows.append({
                'sensor_id': sensor_id,
                'town_name': curr['town_name'],
                'functional_class': curr['functional_class'],
                'year_prev': prev['year'],
                'year_curr': curr['year'],
                'year_diff': year_diff,
                'pop_prev': prev['population'],
                'pop_curr': curr['population'],
                'pop_change': pop_change,
                'traffic_prev': prev['avg_daily_volume'],
                'traffic_curr': curr['avg_daily_volume'],
                'traffic_change': traffic_change,
                'sensor_code': curr['sensor_code'],
                'town_code': curr['town_code'],
                'class_code': curr['class_code']
            })

delta_df = pd.DataFrame(delta_rows)

# Filter sensors with enough delta samples
sensor_counts_delta = delta_df['sensor_id'].value_counts()
valid_sensors_delta = sensor_counts_delta[sensor_counts_delta >= MIN_DATA_POINTS].index
delta_df = delta_df[delta_df['sensor_id'].isin(valid_sensors_delta)]

# Features and target for delta model
features_delta = ['pop_change', 'year_diff', 'sensor_code', 'town_code', 'class_code']
X_delta = delta_df[features_delta]
y_delta = delta_df['traffic_change']

scaler_delta = StandardScaler()
X_delta_scaled = scaler_delta.fit_transform(X_delta)

print(f"Training delta traffic change model on {len(delta_df)} records...")
model_delta = XGBRegressor(n_estimators=100, random_state=42)
model_delta.fit(X_delta_scaled, y_delta)

# --- Validation for absolute volume model ---
print("Validating absolute volume model per sensor...")
results_abs = []
residuals_abs = []

for sensor_id, group in avg_daily_traffic.groupby('sensor_id'):
    X_val_raw = group[features_abs]
    y_true = group['avg_daily_volume']
    X_val_scaled = scaler_abs.transform(X_val_raw)
    y_pred = model_abs.predict(X_val_scaled)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)

    residuals_abs.extend(y_true - y_pred)

    latest_row = group[group['year'] == group['year'].max()].iloc[0]
    modified_input = latest_row[features_abs].copy()
    modified_input['population'] *= 1.05
    baseline_input = scaler_abs.transform([latest_row[features_abs]])
    increased_input = scaler_abs.transform([modified_input])

    baseline_pred = model_abs.predict(baseline_input)[0]
    increased_pred = model_abs.predict(increased_input)[0]
    pct_change = ((increased_pred - baseline_pred) / baseline_pred) * 100 if baseline_pred > 0 else 0

    results_abs.append({
        'sensor_id': sensor_id,
        'town_name': group['town_name'].iloc[0],
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'n_obs': len(group),
        'pct_change_5pct_pop': round(pct_change, 2),
        'baseline_volume': round(baseline_pred, 1),
        'increased_volume': round(increased_pred, 1)
    })

results_abs_df = pd.DataFrame(results_abs)
results_abs_df.to_csv("local_roads_model_results_absolute.csv", index=False)
print("\nSaved absolute model results to local_roads_model_results_absolute.csv")

# --- Validation for delta model ---
print("Validating delta model per sensor...")
results_delta = []
residuals_delta = []

for sensor_id, group in delta_df.groupby('sensor_id'):
    X_val_raw = group[features_delta]
    y_true = group['traffic_change']
    X_val_scaled = scaler_delta.transform(X_val_raw)
    y_pred = model_delta.predict(X_val_scaled)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred)

    residuals_delta.extend(y_true - y_pred)

    # Predict 5% pop increase effect on traffic change
    # For prediction, approximate year_diff as median year_diff for sensor group
    median_year_diff = group['year_diff'].median()
    median_sensor_code = group['sensor_code'].iloc[0]
    median_town_code = group['town_code'].iloc[0]
    median_class_code = group['class_code'].iloc[0]

    baseline_input = scaler_delta.transform([[0, median_year_diff, median_sensor_code, median_town_code, median_class_code]])
    increased_input = scaler_delta.transform([[group['pop_change'].median() * 0.05, median_year_diff, median_sensor_code, median_town_code, median_class_code]])

    baseline_pred = model_delta.predict(baseline_input)[0]
    increased_pred = model_delta.predict(increased_input)[0]
    pct_change = ((increased_pred - baseline_pred) / baseline_pred) * 100 if baseline_pred != 0 else np.nan

    results_delta.append({
        'sensor_id': sensor_id,
        'mae': round(mae, 2),
        'rmse': round(rmse, 2),
        'n_obs': len(group),
        'pct_change_5pct_pop': round(pct_change, 2),
        'baseline_traffic_change': round(baseline_pred, 1),
        'increased_traffic_change': round(increased_pred, 1)
    })

results_delta_df = pd.DataFrame(results_delta)
results_delta_df.to_csv("local_roads_model_results_delta.csv", index=False)
print("Saved delta model results to local_roads_model_results_delta.csv")

# --- Residual analysis ---
sns.histplot(residuals_abs, bins=30, kde=True)
plt.title("Residual Distribution - Absolute Volume Model")
plt.xlabel("Prediction Error")
plt.savefig("residual_distribution_absolute.png")
plt.clf()

sns.histplot(residuals_delta, bins=30, kde=True)
plt.title("Residual Distribution - Delta Model")
plt.xlabel("Prediction Error")
plt.savefig("residual_distribution_delta.png")
plt.clf()

print("Saved residual distribution plots")

# --- Sensor-level trend comparison for absolute model ---
sensor_trends = []

for sensor_id, group in avg_daily_traffic.groupby('sensor_id'):
    if group['year'].nunique() < 2:
        continue
    sorted_group = group.sort_values('year')
    first = sorted_group.iloc[0]
    last = sorted_group.iloc[-1]
    X_first = scaler_abs.transform([first[features_abs]])
    X_last = scaler_abs.transform([last[features_abs]])
    pred_delta = model_abs.predict(X_last)[0] - model_abs.predict(X_first)[0]
    actual_delta = last['avg_daily_volume'] - first['avg_daily_volume']
    sensor_trends.append({
        'sensor_id': sensor_id,
        'town': first['town_name'],
        'actual_change': round(actual_delta),
        'predicted_change': round(pred_delta),
        'error': round(pred_delta - actual_delta)
    })

trend_df = pd.DataFrame(sensor_trends)
trend_df.to_csv("sensor_trend_comparison_absolute.csv", index=False)
print("Saved absolute model sensor trend comparison to sensor_trend_comparison_absolute.csv")

# --- Sensor-level trend comparison for delta model ---
sensor_trends_delta = []

for sensor_id, group in delta_df.groupby('sensor_id'):
    actual_sum = group['traffic_change'].sum()
    predicted_sum = model_delta.predict(scaler_delta.transform(group[features_delta])).sum()
    sensor_trends_delta.append({
        'sensor_id': sensor_id,
        'actual_total_change': round(actual_sum),
        'predicted_total_change': round(predicted_sum),
        'error': round(predicted_sum - actual_sum)
    })

trend_delta_df = pd.DataFrame(sensor_trends_delta)
trend_delta_df.to_csv("sensor_trend_comparison_delta.csv", index=False)
print("Saved delta model sensor trend comparison to sensor_trend_comparison_delta.csv")
breakpoint()