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


# Set up the SQLAlchemy engine and session
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')
Session = sessionmaker(bind=engine)
session = Session()

def plot_residuals(y_test, y_pred, model_name="Model", save_dir="residual_plots"):
    residuals = y_test - y_pred

    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # --- 1. Residuals vs. Predicted ---
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(0, linestyle='--', color='red')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"{model_name} - Residuals vs Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_residuals_vs_predicted_class_7.png"))
    plt.close()

    # --- 2. Histogram of Residuals ---
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residual")
    plt.title(f"{model_name} - Residual Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_residual_distribution_class_7.png"))
    plt.close()

# --- Load traffic sensor metadata ---
print("Loading traffic sensors...")
class_1 = "('(5) Major Collector', '(6) Minor Collector', '(7) Local Road or Street')"
class_2 = "('(3) Other Principal Arterial','(4) Minor Arterial')"
traffic = gpd.read_postgis(
    """
    SELECT location_id as sensor_id, town_name, functional_class, geom
    FROM general_data.traffic_nameplate
    where functional_class in ('(7) Local Road or Street')
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
    where functional_class in ('(7) Local Road or Street')
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
]

X = samples_df[features]
y = samples_df['traffic_pct_change']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

plot_residuals(y_test, y_pred_xgb, model_name="XGBoost")

# Use raw predicted and true percentage changes directly:
y_true = y_test.values  # no expm1
y_pred_true = y_pred_xgb  # no expm1

mae = mean_absolute_error(y_true, y_pred_true)
rmse = np.sqrt(mean_squared_error(y_true, y_pred_true))

mean_target = np.mean(np.abs(y_true))  # use absolute mean to avoid division by near-zero mean
std_target = np.std(y_true)
range_target = np.max(y_true) - np.min(y_true)

normalized_mae_mean = mae / mean_target if mean_target != 0 else np.nan
normalized_mae_std = mae / std_target if std_target != 0 else np.nan
normalized_mae_range = mae / range_target if range_target != 0 else np.nan

normalized_rmse_mean = rmse / mean_target if mean_target != 0 else np.nan
normalized_rmse_std = rmse / std_target if std_target != 0 else np.nan
normalized_rmse_range = rmse / range_target if range_target != 0 else np.nan

print("\nXGBoost Results:")

print(f"MAE: {mae:.4f}")
print(f"Normalized MAE (mean): {normalized_mae_mean:.4f}")
print(f"Normalized MAE (std): {normalized_mae_std:.4f}")
print(f"Normalized MAE (range): {normalized_mae_range:.4f}")

print(f"Raw RMSE: {rmse:.4f}")
print(f"Normalized RMSE (mean): {normalized_rmse_mean:.4f}")
print(f"Normalized RMSE (std): {normalized_rmse_std:.4f}")
print(f"Normalized RMSE (range): {normalized_rmse_range:.4f}")

r2 = r2_score(y_test, y_pred_xgb)
print(f"The r2 score is {r2}")
print("Top Feature Importances:")
print(pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10))

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_true, y=y_pred_xgb, alpha=0.6, s=40)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal Fit')
plt.xlabel('Actual Traffic Volume')
plt.ylabel('Predicted Traffic Volume')
plt.title('Predicted vs. Actual Traffic Volume (XGBoost)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ~~~~PREDICT~~~~~~~

import numpy as np
import pandas as pd

# 1. Determine the max year in population data
max_year = pop_hist['year'].max()

# 2. Get latest population per town at max_year
pop_latest = pop_hist[pop_hist['year'] == max_year][['town_name', 'population']].copy()

# 3. Get latest traffic volume per sensor for max_year
traffic_latest = (
    daily_avg.sort_values(['sensor_id', 'year'], ascending=[True, False])
    .drop_duplicates('sensor_id')
    .rename(columns={'avg_daily_volume': 'traffic_start', 'year': 'traffic_year'})
)
breakpoint()
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

# 8. One-hot encode functional class to match model
functional_dummies = pd.get_dummies(sensor_features_latest['functional_class'], prefix='func_class')

# Ensure all expected columns are present
for col in [c for c in X.columns if c.startswith('func_class_')]:
    if col not in functional_dummies.columns:
        functional_dummies[col] = 0
functional_dummies = functional_dummies[[c for c in X.columns if c.startswith('func_class_')]]

# 9. Assemble final features for prediction
X_pred = pd.concat([
    sensor_features_latest[['pop_pct_change', 'pop_start', 'traffic_start',
                            'log_pop_start', 'log_traffic_start', 'year_gap',
                            'dist_to_mbta_stop', 'mbta_usage']],
    functional_dummies
], axis=1)

# 10. Predict traffic change
predicted_traffic_pct_change = xgb.predict(X_pred)

# 11. Calculate predicted traffic volume
sensor_features_latest['predicted_traffic_pct_change'] = predicted_traffic_pct_change
sensor_features_latest['predicted_traffic_volume'] = sensor_features_latest['traffic_start'] * (1 + predicted_traffic_pct_change)

# 12. Output result
result_df = sensor_features_latest[[
    'sensor_id', 'town_name', 'functional_class', 'traffic_year',
    'pop_start', 'pop_end', 'traffic_start',
    'predicted_traffic_pct_change', 'predicted_traffic_volume'
]]

print(result_df.head())
result_df.to_excel('results.xlsx', index=False)