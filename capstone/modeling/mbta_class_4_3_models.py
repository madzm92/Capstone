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
    evaluate, train_model, get_log_features,multiply_features,get_best_params)


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

#----Join land use------
samples_df = get_land_use_features(land_use, sensor_gdf, samples_df)

# NOTE: log_pop_start high corrolation: removed pop_start
# NOTE: All log distance features performed worse, kept regular distance features
# NOTE: Keep both traffic_start and log_traffic_start since no co-linearity
samples_df = get_log_features(samples_df, ['pop_start','traffic_start'])

samples_df = multiply_features(samples_df)

samples_df = pd.get_dummies(samples_df, columns=['functional_class'], prefix='func_class')


# --- Modeling ---

# --- 3. Update features list ---
features = [
    'pop_pct_change', 'traffic_start', 'dist_to_mbta_stop', 'mbta_usage',
    'log_pop_start', 'log_traffic_start', 'year_gap',
    'dist_to_school_education', 'dist_to_commercial_retail', 'dist_to_transportation',  
    'dist_to_residential_multi-family', 'dist_to_commercial_office', 'dist_to_residential_single_family',
    'dist_to_religious', 'dist_to_recreational_public',  'dist_to_recreational_private', 'dist_to_agricultural',
     'dist_to_industrial', 'dist_to_healthcare', 'dist_to_hotels_hospitality',
    'pop_change_x_dist_to_retail', 'mbta_x_healthcare', 'retail_x_traffic',
       'school_x_pop', 'near_school', 'pop_change_x_mbta',
       'pop_change_x_dist', 'mbta_x_dist'
] + [col for col in samples_df.columns if col.startswith('func_class_')]

# get_best_params(samples_df, features)

# --- XGBoost ---
xgb = XGBRegressor(
    n_estimators=500, 
    learning_rate=0.25, 
    max_depth=6, 
    subsample=0.76, 
    colsample_bytree=0.92, 
    random_state=42,
    min_child_weight=1,
    gamma=0)

oof_preds, oof_true, X_train = train_model(xgb, samples_df, features)

# --- Evaluate ---
evaluate(oof_preds, oof_true, xgb, X_train)

# --- Predict for ALL sensors in functional class 3 and 4 ---

# 1. Determine the max year in population data
common_max_year = min(pop_hist['year'].max(), daily_avg['year'].max())

# 2. Get latest population per town at max_year
pop_latest = pop_hist[pop_hist['year'] == common_max_year][['town_name', 'population']].copy()

traffic_latest = (
    daily_avg[daily_avg['year'] == common_max_year]
    .sort_values(['sensor_id'], ascending=True)
    .drop_duplicates('sensor_id')
    .rename(columns={'avg_daily_volume': 'traffic_start', 'year': 'traffic_year'})
)

# 3. Get latest traffic volume per sensor for max_year
traffic_latest = (
    daily_avg.sort_values(['sensor_id', 'year'], ascending=[True, False])
    .drop_duplicates('sensor_id')
    .rename(columns={'avg_daily_volume': 'traffic_start', 'year': 'traffic_year'})
)

# 3. Merge with traffic and sensor metadata
sensor_features_latest = (
    traffic.merge(traffic_latest, on=['sensor_id','town_name','functional_class'])
           .merge(pop_latest, on='town_name', how='left')
)

# 4. Deduplicate MBTA features — keep closest stop per sensor
sensor_features_dedup = (
    sensor_features.sort_values(['sensor_id', 'dist_to_mbta_stop'])
                   .drop_duplicates('sensor_id', keep='first')
)

# 5. Add MBTA features
sensor_features_latest = sensor_features_latest.merge(
    sensor_features_dedup[['sensor_id', 'mbta_usage', 'dist_to_mbta_stop']],
    on='sensor_id', how='left'
)

# add year predictions are based on
sensor_features_latest['prediction_year'] = sensor_features_latest['traffic_year']

# Fill missing MBTA values
sensor_features_latest['dist_to_mbta_stop'].fillna(sensor_features_latest['dist_to_mbta_stop'].max(), inplace=True)
sensor_features_latest['mbta_usage'].fillna(0, inplace=True)

# 6. Population increase simulation (+5%)
sensor_features_latest['pop_start'] = sensor_features_latest['population']
sensor_features_latest['pop_end'] = sensor_features_latest['pop_start'] * 1.05
sensor_features_latest['pop_pct_change'] = 0.05  # constant for all

# 7. Feature engineering
sensor_features_latest['log_pop_start'] = np.log1p(sensor_features_latest['pop_start'])
sensor_features_latest['log_traffic_start'] = np.log1p(sensor_features_latest['traffic_start'])
sensor_features_latest['year_gap'] = 1  # assuming 1-year projection

sensor_features_latest = get_land_use_features(land_use, sensor_gdf, sensor_features_latest)
sensor_features_latest = multiply_features(sensor_features_latest)
sensor_features_latest = get_log_features(sensor_features_latest, ['dist_to_commercial_retail','dist_to_agricultural','dist_to_hotels_hospitality','dist_to_religious','dist_to_school_education','dist_to_healthcare',])

sensor_features_latest_dummies = pd.get_dummies(sensor_features_latest['functional_class'], columns=['functional_class'], prefix='func_class')
sensor_features_latest = pd.concat([sensor_features_latest,sensor_features_latest_dummies],axis=1)

X_pred = sensor_features_latest[features]

# 10. Predict traffic change
predicted_traffic_pct_change = xgb.predict(X_pred)

# 11. Calculate predicted traffic volume
sensor_features_latest['predicted_traffic_pct_change'] = predicted_traffic_pct_change
sensor_features_latest['predicted_traffic_volume'] = sensor_features_latest['traffic_start'] * (1 + predicted_traffic_pct_change)

# 12. Output result
result_df = sensor_features_latest[[
    'sensor_id', 'town_name', 'functional_class', 
    'pop_start', 'pop_end', 'traffic_start',
    'predicted_traffic_pct_change', 'predicted_traffic_volume', 'prediction_year'
]]
print(result_df.head())

result_df.drop_duplicates(inplace=True, subset=['sensor_id'])
result_df.to_sql('modeling_results', engine, schema='general_data', if_exists='append', index=False)

