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

functional_classes = ['(5) Major Collector', '(6) Minor Collector']
excluded_location_ids = []

# --- Load traffic sensor metadata ---
traffic = load_traffic_sensor_data(engine, functional_classes, excluded_location_ids)

# --- Load historical traffic counts ---
traffic_hist = load_traffic_counts(engine, functional_classes)

# --- Load population data ---
pop_hist = load_pop_data(engine)

# Average daily traffic per sensor
avg_daily = (
    traffic_hist.groupby(['sensor_id', 'date'])['volume'].sum().reset_index()
)
avg_daily['year'] = pd.to_datetime(avg_daily['date']).dt.year

# --- Load land data ---
land_use = load_land_use(engine)

traffic = traffic[traffic.geometry.notnull() & traffic.is_valid & ~traffic.geometry.is_empty]

if traffic.crs is None:
    traffic.set_crs("EPSG:4326", inplace=True)

# Average daily traffic per sensor
avg_daily = (
    traffic_hist.groupby(['sensor_id', 'date'])['volume'].sum().reset_index()
)
avg_daily['year'] = pd.to_datetime(avg_daily['date']).dt.year

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
# samples_df = pd.get_dummies(samples_df, columns=['functional_class'], prefix='func_class')

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

for col in samples_df.columns:
    if col.startswith("dist_to_"):
        samples_df[f"log_{col}"] = np.log1p(samples_df[col])

samples_df["pop_change_x_dist_to_retail"] = samples_df["pop_pct_change"] * samples_df["dist_to_commercial_retail"]
samples_df["mbta_x_healthcare"] = samples_df["dist_to_mbta_stop"] * samples_df["dist_to_healthcare"]
samples_df["near_school"] = (samples_df["dist_to_school_education"] < 0.25).astype(int)
samples_df["near_retail"] = (samples_df["dist_to_commercial_retail"] < 0.25).astype(int)
samples_df["retail_x_traffic"] = samples_df["dist_to_commercial_retail"] * samples_df["traffic_start"]
samples_df["school_x_pop"] = samples_df["dist_to_school_education"] * samples_df["pop_start"]

features = [
    'pop_pct_change',
    'retail_x_traffic',
    'traffic_start',
    'pop_start',
    'year_gap',
    'mbta_x_healthcare',
    'log_dist_to_commercial_retail',
    'pop_change_x_dist',
    'log_dist_to_agricultural',
    'log_dist_to_hotels_hospitality',
    'log_dist_to_religious',
    'log_dist_to_school_education',
    'log_dist_to_healthcare',
    'school_x_pop'
]


# --- XGBoost ---
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)

oof_preds, oof_true, X_train = train_model(xgb, samples_df, features)

# --- Evaluate ---
evaluate(oof_true, oof_preds, xgb, X_train)
breakpoint()

# ~~~~PREDICT~~~~~~~

# Features must be the same as model training:
features = [
    'pop_pct_change',
    'retail_x_traffic',
    'traffic_start',
    'pop_start',
    'year_gap',
    'mbta_x_healthcare',
    'log_dist_to_commercial_retail',
    'pop_change_x_dist',
    'log_dist_to_agricultural',
    'log_dist_to_hotels_hospitality',
    'log_dist_to_religious',
    'log_dist_to_school_education',
    'log_dist_to_healthcare',
    'school_x_pop'
]

# If you want to predict traffic changes for a 5% increase in population, set pop_pct_change = 0.05
samples_df['pop_pct_change'] = 0.05

# Recalculate any dependent features that involve pop_pct_change
samples_df['pop_change_x_dist'] = samples_df['pop_pct_change'] * samples_df['dist_to_mbta_stop']
samples_df['pop_change_x_dist_to_retail'] = samples_df['pop_pct_change'] * samples_df['dist_to_commercial_retail']

# Recalculate interaction features if needed
samples_df['retail_x_traffic'] = samples_df['dist_to_commercial_retail'] * samples_df['traffic_start']
samples_df['mbta_x_healthcare'] = samples_df['dist_to_mbta_stop'] * samples_df['dist_to_healthcare']
samples_df['school_x_pop'] = samples_df['dist_to_school_education'] * samples_df['pop_start']

# Make sure log features are present
for col in ['dist_to_commercial_retail', 'dist_to_agricultural', 'dist_to_hotels_hospitality', 
            'dist_to_religious', 'dist_to_school_education', 'dist_to_healthcare']:
    log_col = 'log_' + col
    if log_col not in samples_df.columns:
        samples_df[log_col] = np.log1p(samples_df[col])

# Also ensure year_gap is set properly (usually 1 for predicting next year)
samples_df['year_gap'] = 1

# --- Step 2: Extract features matrix for prediction ---
X_pred = samples_df[features]

# --- Step 3: Make predictions ---
# Predict log-scale traffic % change
y_pred_log = xgb.predict(X_pred.values)

# Convert back to percentage change scale (inverse of log1p)
epsilon = 1e-4  # same small number you added during training
samples_df['predicted_traffic_pct_change'] = np.expm1(y_pred_log) - epsilon

# --- Step 4: Calculate predicted traffic volume after population increase ---
samples_df['predicted_traffic_volume'] = samples_df['traffic_start'] * (1 + samples_df['predicted_traffic_pct_change'])

# --- Step 5: Output relevant results ---
result_cols = [
    'sensor_id', 'town_name', 'functional_class','pop_start','pop_end', 'traffic_start',
    'pop_pct_change', 'predicted_traffic_pct_change', 'predicted_traffic_volume',
]

result_df = samples_df[result_cols].copy()
result_df = result_df.drop_duplicates(keep='last')

print(result_df.head())

# Save results if desired
breakpoint()
result_df.drop_duplicates(inplace=True, subset=['sensor_id'])
result_df.drop('pop_pct_change', axis=1,inplace=True)
result_df.to_sql('modeling_results', engine, schema='general_data', if_exists='append', index=False)
