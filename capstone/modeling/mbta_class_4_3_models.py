import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import KFold
from capstone.modeling.shared_functions import (
    get_distance_to_category, plot_residuals, 
    plot_diff, show_counts, load_traffic_sensor_data, 
    load_traffic_counts, load_pop_data, load_land_use, 
    get_mbta_data, get_land_use_features, get_extra_features,
    evaluate, train_model)


# Set up the SQLAlchemy engine and session
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')
Session = sessionmaker(bind=engine)
session = Session()


functional_classes = ['(4) Minor Arterial', '(3) Other Principal Arterial']
excluded_location_ids = ['3083']

# --- Load traffic sensor metadata ---
traffic = load_traffic_sensor_data(engine, functional_classes, excluded_location_ids)

# --- Load historical traffic counts ---
traffic_hist = load_traffic_counts(engine, functional_classes, excluded_location_ids)

# --- Load population data ---
pop_hist = load_pop_data(engine)

# --- Load land use data ---
land_use = load_land_use(engine)

# Average daily traffic per sensor
avg_daily = (
    traffic_hist.groupby(['sensor_id', 'date'])['volume'].sum().reset_index()
)
avg_daily['year'] = pd.to_datetime(avg_daily['date']).dt.year

traffic = traffic[traffic.geometry.notnull() & traffic.is_valid & ~traffic.geometry.is_empty]

# show sensor counts for slide 1
# show_counts(avg_daily)

daily_avg = (
    avg_daily.groupby(['sensor_id', 'year'])['volume'].mean().reset_index()
    .rename(columns={'volume': 'avg_daily_volume'})
)

# --- Filter to valid sensors ---
sensor_years = daily_avg.groupby('sensor_id')['year'].agg(['min', 'max', 'count'])
valid_sensors = sensor_years[(sensor_years['count'] >= 2) & ((sensor_years['max'] - sensor_years['min']) >= 1)].index
daily_avg = daily_avg[daily_avg['sensor_id'].isin(valid_sensors)]

# --- Merge with population ---
pop_hist['year'] = pop_hist['year'].astype(int)
daily_avg = daily_avg.merge(
    traffic[['sensor_id', 'town_name', 'functional_class']], on='sensor_id', how='left'
)
samples = daily_avg.merge(pop_hist, on=['town_name', 'year'])

# --- Create time-paired samples ---
print("Creating time-difference samples...")
samples = samples.sort_values(['sensor_id', 'year'])

samples_df = (
    samples.sort_values(['sensor_id', 'year'])
    .groupby('sensor_id')
    .agg({
        'year': ['first', 'last'],
        'avg_daily_volume': ['first', 'last'],
        'population': ['first', 'last'],
        'town_name': 'first',
        'functional_class': 'first'
    })
)

samples_df.columns = [
    'year_start', 'year_end',
    'traffic_start', 'traffic_end',
    'pop_start', 'pop_end',
    'town_name', 'functional_class'
]
samples_df = samples_df.reset_index()

# Recalculate features
samples_df = get_extra_features(samples_df)

# --- Add MBTA stop usage and distance ---
mbta_stops = get_mbta_data(engine)

print("Calculating distance to nearest MBTA stop...")
sensor_gdf = traffic[['sensor_id', 'geom']].copy()
sensor_gdf = sensor_gdf.set_geometry('geom')
distance_join = gpd.sjoin_nearest(sensor_gdf, mbta_stops[['stop_id', 'mbta_usage', 'geom']], distance_col='dist_to_mbta_stop')

sensor_features = distance_join[['sensor_id', 'dist_to_mbta_stop', 'mbta_usage']]
samples_df = samples_df.merge(sensor_features, on='sensor_id', how='left')
samples_df.fillna(0, inplace=True)

# --- One-hot encode functional_class ---
samples_df = pd.get_dummies(samples_df, columns=['functional_class'], prefix='func_class')

#----Join land use------
samples_df = get_land_use_features(land_use, sensor_gdf, samples_df)

# --- Modeling ---

# --- 1. Log-transform the target ---
# Add small epsilon to handle near-zero or negative percentage changes
epsilon = 1e-4
samples_df['log_traffic_pct_change'] = np.log1p(samples_df['traffic_pct_change'] + epsilon)

# --- 2. Add interaction features ---
samples_df['pop_change_x_mbta'] = samples_df['pop_pct_change'] * samples_df['mbta_usage']
samples_df['pop_change_x_dist'] = samples_df['pop_pct_change'] * samples_df['dist_to_mbta_stop']
samples_df['mbta_x_dist'] = samples_df['mbta_usage'] * samples_df['dist_to_mbta_stop']


# --- 3. Update features list ---
features = [
    'pop_pct_change', 'pop_start', 'traffic_start',
    'log_pop_start', 'log_traffic_start', 'year_gap',
    'dist_to_mbta_stop', 'mbta_usage',
    'pop_change_x_mbta', 'pop_change_x_dist', 'mbta_x_dist', 'dist_to_school_education', 'dist_to_commercial_retail', 
    'dist_to_transportation',  'dist_to_residential_multi-family', 'dist_to_commercial_office', 'dist_to_residential_single_family',
    'dist_to_religious', 'dist_to_recreational_public',  'dist_to_recreational_private', 'dist_to_agricultural',
     'dist_to_industrial', 'dist_to_healthcare', 'dist_to_hotels_hospitality'
] + [col for col in samples_df.columns if col.startswith('func_class_')]

# --- XGBoost ---
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)

oof_preds, oof_true, X_train = train_model(xgb, samples_df, features)

# --- Evaluate ---
evaluate(oof_preds, oof_true, xgb, X_train)

breakpoint()

#~~~~Predictions
# --- Predict for ALL sensors in functional class 3 and 4 ---
print("Generating predictions for all valid sensor locations...")

# Start from original traffic metadata
all_sensors = traffic[['sensor_id', 'functional_class', 'town_name']].drop_duplicates()

# Merge with known starting traffic/population
predict_df = all_sensors.merge(
    samples_df[['sensor_id', 'traffic_start', 'pop_start', 'log_pop_start', 'log_traffic_start']],
    on='sensor_id', how='left'
)

predict_df = get_land_use_features(land_use, sensor_gdf, predict_df)

# Fill missing values with median or default assumptions
predict_df['traffic_start'] = predict_df['traffic_start'].fillna(predict_df['traffic_start'].median())
predict_df['pop_start'] = predict_df['pop_start'].fillna(predict_df['pop_start'].median())
predict_df['log_pop_start'] = np.log1p(predict_df['pop_start'])
predict_df['log_traffic_start'] = np.log1p(predict_df['traffic_start'])

# Set population increase assumption (e.g., 5%)
predict_df['pop_pct_change'] = 0.05
predict_df['year_gap'] = 1
predict_df['pop_end'] = predict_df['pop_start'] * (1 + predict_df['pop_pct_change'])


# --- Add distance to MBTA stop and usage ---
predict_df = predict_df.merge(sensor_features, on='sensor_id', how='left')
predict_df['dist_to_mbta_stop'] = predict_df['dist_to_mbta_stop'].fillna(predict_df['dist_to_mbta_stop'].median())
predict_df['mbta_usage'] = predict_df['mbta_usage'].fillna(0)

# --- Add interaction terms ---
predict_df['pop_change_x_mbta'] = predict_df['pop_pct_change'] * predict_df['mbta_usage']
predict_df['pop_change_x_dist'] = predict_df['pop_pct_change'] * predict_df['dist_to_mbta_stop']
predict_df['mbta_x_dist'] = predict_df['mbta_usage'] * predict_df['dist_to_mbta_stop']

# --- One-hot encode functional class ---
functional_class_col = predict_df[['sensor_id', 'functional_class']].copy()
predict_df = pd.get_dummies(predict_df, columns=['functional_class'], prefix='func_class')

# 1. Merge in year_start for each sensor (if available)
sensor_years = samples_df[['sensor_id', 'year_start']].drop_duplicates()
predict_df = predict_df.merge(sensor_years, on='sensor_id', how='left')

# 2. Fill missing year_start with default (e.g., latest year with data)
predict_df['year_start'] = predict_df['year_start'].fillna(2023).astype(int)

# 3. Compute traffic_year (the year we're forecasting for)
predict_df['traffic_year'] = predict_df['year_start'] + predict_df['year_gap']

# 4. Ensure all expected dummy columns are present
for col in [c for c in samples_df.columns if c.startswith("func_class_")]:
    if col not in predict_df.columns:
        predict_df[col] = 0
# --- Final prediction ---
X_pred_all = predict_df[features]
y_pred_log_all = xgb.predict(X_pred_all.values)
predict_df['predicted_traffic_pct_change'] = np.expm1(y_pred_log_all) - epsilon
predict_df['predicted_traffic_volume'] = predict_df['traffic_start'] * (1 + predict_df['predicted_traffic_pct_change'])

# --- Save Results ---
output_cols = [
    'sensor_id', 'town_name', 'pop_start', 'traffic_start', 'dist_to_mbta_stop', 'mbta_usage',
    'predicted_traffic_pct_change', 'predicted_traffic_volume'
] + [c for c in predict_df.columns if c.startswith('func_class_')]

results_full = predict_df[output_cols].copy()

print(results_full.head())

# Drop duplicates by sensor, overwrite old predictions
results_full.drop_duplicates(subset=['sensor_id'], inplace=True)

results_full = predict_df[[
    'sensor_id', 'town_name', 'pop_start', 'traffic_start', 'predicted_traffic_pct_change', 
    'predicted_traffic_volume', "traffic_year", "pop_end"
]].copy()

# Join functional class back in
results_full = results_full.merge(functional_class_col, on='sensor_id', how='left')
breakpoint()
results_full.drop_duplicates(inplace=True, subset=['sensor_id'])
results_full.drop('traffic_year', axis=1,inplace=True)
results_full.to_sql('modeling_results', engine, schema='general_data', if_exists='append', index=False)

# WHY ARE THERE DUPLICATES?!?!?
