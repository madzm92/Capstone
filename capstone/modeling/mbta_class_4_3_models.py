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
    plt.savefig(os.path.join(save_dir, f"{model_name}_residuals_vs_predicted_class_4_3.png"))
    plt.close()

    # --- 2. Histogram of Residuals ---
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residual")
    plt.title(f"{model_name} - Residual Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_residual_distribution_class_4_3.png"))
    plt.close()

# --- Load traffic sensor metadata ---
print("Loading traffic sensors...")

traffic = gpd.read_postgis(
    """
    SELECT location_id as sensor_id, town_name, functional_class, geom
    FROM general_data.traffic_nameplate
    where functional_class in ('(4) Minor Arterial', '(3) Other Principal Arterial')
    and location_id != '3083'
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
    where functional_class in ('(4) Minor Arterial', '(3) Other Principal Arterial')
    and tn.location_id != '3083'
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
    'pop_change_x_mbta', 'pop_change_x_dist', 'mbta_x_dist'
] + [col for col in samples_df.columns if col.startswith('func_class_')]

X = samples_df[features]
y = samples_df['log_traffic_pct_change']

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Linear Regression ---
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred_log = model.predict(X_test)
# y_pred = np.expm1(y_pred_log) - epsilon
y_true = np.expm1(y_test) - epsilon

# print("\nLinear Regression:")
# print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
# print(f"The r2 score is {r2_score(y_true, y_pred)}")

# --- Random Forest ---
# rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, oob_score=True)
# rf.fit(X_train, y_train)
# y_pred_log_rf = rf.predict(X_test)
# y_pred_rf = np.expm1(y_pred_log_rf) - epsilon

# print("\nRandom Forest:")
# print(f"MAE: {mean_absolute_error(y_true, y_pred_rf):.4f}, RMSE: {np.sqrt(mean_squared_error(y_true, y_pred_rf)):.4f}, OOB Score: {rf.oob_score_:.4f}")
# print(f"The r2 score is {r2_score(y_true, y_pred_rf)}")
# print("Top Feature Importances:")
# print(pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10))

# --- XGBoost ---
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb.fit(X_train, y_train)
y_pred_log_xgb = xgb.predict(X_test)
y_pred_xgb = np.expm1(y_pred_log_xgb) - epsilon

print("\nXGBoost Results:")
print(f"MAE: {mean_absolute_error(y_true, y_pred_xgb):.4f}, RMSE: {np.sqrt(mean_squared_error(y_true, y_pred_xgb)):.4f}")
print(f"The r2 score is {r2_score(y_true, y_pred_xgb)}")
print("Top Feature Importances:")
print(pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10))

# --- Plot Residuals ---
# plot_residuals(y_true, y_pred, model_name="Linear Regression")
# plot_residuals(y_true, y_pred_rf, model_name="Random Forest")
plot_residuals(y_true, y_pred_xgb, model_name="XGBoost")

mae = mean_absolute_error(y_true, y_pred_xgb)
mean_target = np.mean(y_true)
std_target = np.std(y_true)
range_target = np.max(y_true) - np.min(y_true)

normalized_mae_mean = mae / mean_target
normalized_mae_std = mae / std_target
normalized_mae_range = mae / range_target

print(f"MAE: {mae:.4f}")
print(f"Normalized MAE (mean): {normalized_mae_mean:.4f}")
print(f"Normalized MAE (std): {normalized_mae_std:.4f}")
print(f"Normalized MAE (range): {normalized_mae_range:.4f}")

rmse = np.sqrt(np.mean((y_true - y_pred_xgb) ** 2))
mean_y = np.mean(y_true)
std_y = np.std(y_true)
range_y = np.max(y_true) - np.min(y_true)

rmse_norm_mean = rmse / mean_y
rmse_norm_std = rmse / std_y
rmse_norm_range = rmse / range_y

print("Raw RMSE:", rmse)
print("Normalized RMSE (mean):", rmse_norm_mean)
print("Normalized RMSE (std):", rmse_norm_std)
print("Normalized RMSE (range):", rmse_norm_range)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_true, y=y_pred_xgb, alpha=0.6, s=40)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal Fit')
plt.xlabel('Actual Traffic Volume % Change')
plt.ylabel('Predicted Traffic Volume % Change')
plt.title('Predicted vs. Actual Traffic Volume (XGBoost)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# import shap
# explainer = shap.Explainer(xgb)
# shap_values = explainer(X_test)
# shap.plots.beeswarm(shap_values)



# --- Diagnose Class 4 behavior ---
class4_df = samples_df.copy()

# Compute raw prediction errors if available
if 'log_traffic_pct_change' in class4_df.columns:
    class4_df['actual'] = np.expm1(class4_df['log_traffic_pct_change']) - epsilon
    # If you're doing in-sample diagnostics (optional)
    class4_features = class4_df[features]
    class4_preds_log = xgb.predict(class4_features)
    class4_df['predicted'] = np.expm1(class4_preds_log) - epsilon
    class4_df['abs_error'] = np.abs(class4_df['predicted'] - class4_df['actual'])

    top_outliers = class4_df.sort_values('abs_error', ascending=False).head(10)
    print("\nTop 10 Class 4 Outliers:")
    print(top_outliers[['sensor_id', 'traffic_start', 'traffic_end', 'actual', 'predicted', 'abs_error']])

# Visualize actual vs. predicted
# plt.figure(figsize=(8, 6))
# plt.scatter(class4_df['actual'], class4_df['predicted'], alpha=0.6, s=40)
# plt.plot([class4_df['actual'].min(), class4_df['actual'].max()],
#          [class4_df['actual'].min(), class4_df['actual'].max()],
#          'r--')
# plt.xlabel("Actual Traffic Change (Class 4)")
# plt.ylabel("Predicted Traffic Change")
# plt.title("Actual vs Predicted Traffic % Change — Class 4 Roads")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Check for large actual and near-zero predicted
# extreme = class4_df[(class4_df['actual'] > 300) & (class4_df['predicted'] < 1)]
# print("\nExtreme Class 4 Outliers:")
# print(extreme[['sensor_id', 'traffic_start', 'traffic_end', 'pop_start', 'pop_pct_change',
#                'year_gap', 'dist_to_mbta_stop', 'mbta_usage', 'actual', 'predicted']])
