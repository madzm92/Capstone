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

# Set up the SQLAlchemy engine and session
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')
Session = sessionmaker(bind=engine)
session = Session()

# --- Load traffic sensor metadata ---
print("Loading traffic sensors...")
traffic = gpd.read_postgis(
    """
    SELECT location_id as sensor_id, town_name, functional_class, geom
    FROM general_data.traffic_nameplate
    """,
    engine,
    geom_col="geom"
)

# --- Load historical traffic counts ---
print("Loading traffic counts...")
traffic_hist = pd.read_sql("""
    SELECT tn.location_id as sensor_id, tc.start_date_time as timestamp, tc.hourly_count as volume
    FROM general_data.traffic_nameplate tn  
    LEFT JOIN general_data.traffic_counts tc ON tn.location_id = tc.location_id
""", engine)

# --- Load population data ---
print("Loading population data...")
pop_hist = pd.read_sql("""
    SELECT ap.year, tcc.town_name, ap.total_population as population
    FROM general_data.annual_population ap
    LEFT JOIN general_data.town_census_crosswalk tcc ON ap.zip_code = tcc.zip_code
""", engine)

# --- Process traffic data ---
print("Processing traffic data...")
traffic_hist['timestamp'] = pd.to_datetime(traffic_hist['timestamp'])
traffic_hist['date'] = traffic_hist['timestamp'].dt.date
traffic_hist['year'] = traffic_hist['timestamp'].dt.year

# Average daily traffic per sensor
avg_daily = (
    traffic_hist.groupby(['sensor_id', 'date'])['volume'].sum().reset_index()
)
avg_daily['year'] = pd.to_datetime(avg_daily['date']).dt.year

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
samples_df = []

for sensor_id, group in samples.groupby('sensor_id'):
    group = group.sort_values('year')
    for i in range(len(group) - 1):
        samples_df.append({
            'sensor_id': sensor_id,
            'year_start': group.iloc[i]['year'],
            'year_end': group.iloc[i+1]['year'],
            'traffic_start': group.iloc[i]['avg_daily_volume'],
            'traffic_end': group.iloc[i+1]['avg_daily_volume'],
            'pop_start': group.iloc[i]['population'],
            'pop_end': group.iloc[i+1]['population'],
            'town_name': group.iloc[i]['town_name'],
            'functional_class': group.iloc[i]['functional_class']
        })

samples_df = pd.DataFrame(samples_df)
samples_df['traffic_pct_change'] = (samples_df['traffic_end'] - samples_df['traffic_start']) / samples_df['traffic_start']
samples_df['pop_pct_change'] = (samples_df['pop_end'] - samples_df['pop_start']) / samples_df['pop_start']
samples_df['log_pop_start'] = np.log1p(samples_df['pop_start'])
samples_df['log_traffic_start'] = np.log1p(samples_df['traffic_start'])
samples_df['year_gap'] = samples_df['year_end'] - samples_df['year_start']

# --- Add MBTA stop usage and distance ---
print("Loading MBTA stop data...")
mbta_stops = gpd.read_postgis(
    """
    SELECT id as stop_id, stop_name, town_name, geometry as geom
    FROM general_data.commuter_rail_stops
    """,
    engine,
    geom_col="geom"
)

print("Loading MBTA trip data for 2018...")
mbta_trips = pd.read_sql("""
    SELECT stop_id,stop_datetime, direction_id as direction, average_ons, average_offs
    FROM general_data.commuter_rail_trips
""", engine)

mbta_trips['year'] = mbta_trips['stop_datetime'].dt.year.replace({2024: 2023})  # Treat Jan 2024 as 2023

mbta_agg = (mbta_trips.groupby(['stop_id', 'year'])[['average_ons', 'average_offs']].sum().reset_index())
mbta_agg['mbta_usage'] = mbta_agg['average_ons'] + mbta_agg['average_offs']

mbta_stops = mbta_stops.merge(mbta_agg[['stop_id', 'mbta_usage']], on='stop_id', how='left')

# Prepare geometries
traffic = traffic[traffic.geometry.notnull() & traffic.is_valid & ~traffic.geometry.is_empty]
mbta_stops = mbta_stops[mbta_stops.geometry.notnull() & mbta_stops.is_valid & ~mbta_stops.geometry.is_empty]

mbta_stops.set_crs("EPSG:26986", inplace=True, allow_override=True)

if traffic.crs.to_epsg() != 26986:
    traffic = traffic.to_crs("EPSG:26986")

print("Calculating distance to nearest MBTA stop...")
sensor_gdf = traffic[['sensor_id', 'geom']].copy()
sensor_gdf = sensor_gdf.set_geometry('geom')
distance_join = gpd.sjoin_nearest(sensor_gdf, mbta_stops[['stop_id', 'mbta_usage', 'geom']], distance_col='dist_to_mbta_stop')

sensor_features = distance_join[['sensor_id', 'dist_to_mbta_stop', 'mbta_usage']]
samples_df = samples_df.merge(sensor_features, on='sensor_id', how='left')
samples_df.fillna(0, inplace=True)

# --- One-hot encode functional_class ---
samples_df = pd.get_dummies(samples_df, columns=['functional_class'], prefix='func_class')

# --- Modeling ---
print("Running models...")
features = [
    'pop_pct_change', 'pop_start', 'traffic_start',
    'log_pop_start', 'log_traffic_start', 'year_gap',
    'dist_to_mbta_stop', 'mbta_usage'
] + [col for col in samples_df.columns if col.startswith('func_class_')]

X = samples_df[features]
y = samples_df['traffic_pct_change']

# Linear Regression
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)

# print("\nLinear Regression:")
# print(f"MAE: {mean_absolute_error(y_test, y_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# Random Forest
rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, oob_score=True)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest:")
print(f"MAE: {mean_absolute_error(y_test, y_pred_rf):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_rf)):.4f}, OOB Score: {rf.oob_score_:.4f}")
print("Top Feature Importances:")
print(pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10))

# # XGBoost
# xgb = XGBRegressor(
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=4,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42
# )
# xgb.fit(X_train, y_train)
# y_pred_xgb = xgb.predict(X_test)
# print("\nXGBoost Results:")
# print(f"MAE: {mean_absolute_error(y_test, y_pred_xgb):.4f}, RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_xgb)):.4f}")
# print("Top Feature Importances:")
# print(pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10))

# ~~~~PREDICT

import numpy as np
import pandas as pd

# 1. Determine the max year in population data
max_year = pop_hist['year'].max()

# 2. Get latest population per town at max_year
pop_latest = pop_hist[pop_hist['year'] == max_year][['town_name', 'population']].copy()

# 3. Get latest traffic volume per sensor for max_year
traffic_latest = daily_avg[daily_avg['year'] == max_year][['sensor_id', 'avg_daily_volume']].copy()
traffic_latest = traffic_latest.rename(columns={'avg_daily_volume': 'traffic_start'})

# 4. Merge sensor metadata (functional_class, town_name) and MBTA features
sensor_features_latest = traffic.merge(traffic_latest, on='sensor_id').merge(
    pop_latest, on='town_name', how='left'
)

# 5. Add MBTA features: dist_to_mbta_stop, mbta_usage from sensor_features (from your previous code)
sensor_features_latest = sensor_features_latest.merge(
    sensor_features, on='sensor_id', how='left'
)

sensor_features_latest.fillna({'dist_to_mbta_stop': sensor_features_latest['dist_to_mbta_stop'].max(), 'mbta_usage': 0}, inplace=True)

# 6. Calculate new population after +5% increase
sensor_features_latest['pop_start'] = sensor_features_latest['population']
sensor_features_latest['pop_end'] = sensor_features_latest['pop_start'] * 1.05
sensor_features_latest['pop_pct_change'] = 0.05  # fixed increase

# 7. Log transform features
sensor_features_latest['log_pop_start'] = np.log1p(sensor_features_latest['pop_start'])
sensor_features_latest['log_traffic_start'] = np.log1p(sensor_features_latest['traffic_start'])

# 8. year_gap = 1 (predicting one year difference)
sensor_features_latest['year_gap'] = 1

# 9. One-hot encode functional_class to match model features
functional_dummies = pd.get_dummies(sensor_features_latest['functional_class'], prefix='func_class')

# To ensure same dummy columns as training data
for col in [c for c in X.columns if c.startswith('func_class_')]:
    if col not in functional_dummies.columns:
        functional_dummies[col] = 0

# Align columns order
functional_dummies = functional_dummies[[c for c in X.columns if c.startswith('func_class_')]]

# 10. Assemble feature DataFrame for prediction
X_pred = pd.concat([
    sensor_features_latest[['pop_pct_change', 'pop_start', 'traffic_start', 'log_pop_start', 'log_traffic_start', 'year_gap', 'dist_to_mbta_stop', 'mbta_usage']],
    functional_dummies
], axis=1)

# 11. Predict traffic_pct_change with Random Forest model
predicted_traffic_pct_change = rf.predict(X_pred)

# 12. Calculate predicted traffic volume after population increase
sensor_features_latest['predicted_traffic_pct_change'] = predicted_traffic_pct_change
sensor_features_latest['predicted_traffic_volume'] = sensor_features_latest['traffic_start'] * (1 + predicted_traffic_pct_change)

# 13. Output results
result_df = sensor_features_latest[[
    'sensor_id', 'town_name', 'functional_class', 'pop_start', 'pop_end',
    'traffic_start', 'predicted_traffic_pct_change', 'predicted_traffic_volume'
]]

print(result_df.head())
result_df.to_excel('results.xlsx')


breakpoint()