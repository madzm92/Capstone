import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sqlalchemy.orm import sessionmaker
from xgboost import XGBRegressor


# initial model that includes features:
# - population
# - land use
# Tests model With
# - Regression
# - Random Forest Regression
# - XGBoost

# Info Learned
# Class 5 did better than class 7
# Land use seems to have no impact on the model

# Next Steps on new file
# Remove land use
# add mbta stop distance as a feature
# combine classes and use as a feature
# test other models

# Set up the SQLAlchemy engine and session
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')
Session = sessionmaker(bind=engine)
session = Session()

land_use_weights = {
    'Residential: Single Family': 0.3,
    'Residential: Multi-Family': 0.5,
    'School/Education': 0.6,
    'Healthcare': 0.4,
    'Institutional/Charitable': 0.3,
    'Commercial: Retail': 0.4,
    'Commercial: Office': 0.2,
    'Hotels/Hospitality': 0.3,
    'Industrial': 0.1,
    'Utilities': 0.05,
    'Transportation': 0.05,
    'Recreational: Public': 0.3,
    'Recreational: Private': 0.2,
    'Religious': 0.2,
    'Government': 0.25,
    'Mixed Use': 0.35,
    'Vacant': 0.0,
    'Other': 0.0
}



# --- Load sensor, population, and traffic data ---
print("Loading base data...")
traffic = gpd.read_postgis(
    """
    SELECT location_id as sensor_id, town_name, functional_class, geom
    FROM general_data.traffic_nameplate
    WHERE functional_class = '(5) Major Collector'
    """,
    engine,
    geom_col="geom"
)

traffic_hist = pd.read_sql("""
    SELECT tn.location_id as sensor_id, tc.start_date_time as timestamp, tc.hourly_count as volume
    FROM general_data.traffic_nameplate tn  
    LEFT JOIN general_data.traffic_counts tc ON tn.location_id = tc.location_id
    WHERE tn.functional_class = '(5) Major Collector'
""", engine)

pop_hist = pd.read_sql("""
    SELECT ap.year, tcc.town_name, ap.total_population as population
    FROM general_data.annual_population ap
    LEFT JOIN general_data.town_census_crosswalk tcc ON ap.zip_code = tcc.zip_code
""", engine)

# --- Process traffic data to daily averages ---
print("Aggregating traffic data...")
traffic_hist['timestamp'] = pd.to_datetime(traffic_hist['timestamp'])
traffic_hist['date'] = traffic_hist['timestamp'].dt.date
traffic_hist['year'] = traffic_hist['timestamp'].dt.year

# Compute average daily traffic volume per year per sensor
daily_avg = (
    traffic_hist.groupby(['sensor_id', 'date'])['volume'].sum().reset_index()
)
daily_avg['year'] = pd.to_datetime(daily_avg['date']).dt.year

# Now compute yearly average of daily traffic
daily_avg = (
    daily_avg.groupby(['sensor_id', 'year'])['volume'].mean().reset_index()
    .rename(columns={'volume': 'avg_daily_volume'})
)
daily_avg.columns = ['sensor_id', 'year', 'avg_daily_volume']

# --- Filter to valid sensors ---
print("Filtering sensors with sufficient data...")
sensor_years = daily_avg.groupby('sensor_id')['year'].agg(['min', 'max', 'count'])
valid_sensors = sensor_years[(sensor_years['count'] >= 2) & ((sensor_years['max'] - sensor_years['min']) >= 1)].index
daily_avg = daily_avg[daily_avg['sensor_id'].isin(valid_sensors)]

# --- Join population data ---
pop_hist['year'] = pop_hist['year'].astype(int)
daily_avg = daily_avg.merge(
    traffic[['sensor_id', 'town_name']], on='sensor_id', how='left'
)
samples = daily_avg.merge(pop_hist, on=['town_name', 'year'])

# --- Create model samples by year pairs ---
print("Preparing samples for modeling...")
samples = samples.sort_values(['sensor_id', 'year'])
samples_df = []

for sensor_id, group in samples.groupby('sensor_id'):
    group = group.sort_values('year')
    for i in range(len(group) - 1):
        row = {
            'sensor_id': sensor_id,
            'year_start': group.iloc[i]['year'],
            'year_end': group.iloc[i+1]['year'],
            'traffic_start': group.iloc[i]['avg_daily_volume'],
            'traffic_end': group.iloc[i+1]['avg_daily_volume'],
            'pop_start': group.iloc[i]['population'],
            'pop_end': group.iloc[i+1]['population'],
            'town_name': group.iloc[i]['town_name']
        }
        samples_df.append(row)

samples_df = pd.DataFrame(samples_df)

samples_df['traffic_pct_change'] = (samples_df['traffic_end'] - samples_df['traffic_start']) / samples_df['traffic_start']
samples_df['pop_pct_change'] = (samples_df['pop_end'] - samples_df['pop_start']) / samples_df['pop_start']
samples_df['log_pop_start'] = np.log1p(samples_df['pop_start'])
samples_df['log_traffic_start'] = np.log1p(samples_df['traffic_start'])
samples_df['year_gap'] = samples_df['year_end'] - samples_df['year_start']

# --- Load land use and grouped mapping ---
print("Loading and preparing land use data...")
land_use = gpd.read_postgis(
    """
    SELECT geometry, use_type, town_name
    FROM general_data.shapefiles
    """,
    engine,
    geom_col="geometry"
)

grouped_map = pd.read_csv("grouped_land_use_types.csv")  # contains use_type and grouped
land_use = land_use.merge(grouped_map, left_on='use_type',right_on='original', how='left')

# --- Geometry cleaning and reprojection ---
print("Preparing geometries...")
land_use = land_use[land_use.geometry.notnull() & land_use.is_valid & ~land_use.geometry.is_empty]
traffic = traffic[traffic.geometry.notnull() & traffic.is_valid & ~traffic.geometry.is_empty]

if land_use.crs is None:
    land_use.set_crs("EPSG:26986", inplace=True)

if traffic.crs is None:
    traffic.set_crs("EPSG:4326", inplace=True)

target_crs = "EPSG:26986"
land_use = land_use.to_crs(target_crs)
traffic = traffic.to_crs(target_crs)

# --- Spatial Join: Count grouped land use types around each sensor ---
print("Computing land use context around each sensor...")
sensor_buffer = traffic[['sensor_id', 'geom']].copy()
sensor_buffer = sensor_buffer.set_geometry('geom')
sensor_buffer['geometry'] = sensor_buffer.buffer(500)

sensor_landuse = gpd.sjoin_nearest(sensor_buffer, land_use, how='left', distance_col='dist_to_landuse')
landuse_counts = (
    sensor_landuse.groupby(['sensor_id', 'grouped']).size()
    .unstack(fill_value=0)
    .reset_index()
)
# Assume landuse_counts is a DataFrame where columns are land use group names and values are counts or presence (0/1)

# Multiply each column by its weight
for col in landuse_counts.columns:
    # convert column to numeric, coerce errors to NaN, then fill NaN with 0
    landuse_counts[col] = pd.to_numeric(landuse_counts[col], errors='coerce').fillna(0)
    
    if col in land_use_weights:
        landuse_counts[col + "_weighted"] = landuse_counts[col] * land_use_weights[col]
    else:
        landuse_counts[col + "_weighted"] = landuse_counts[col] * land_use_weights["Other"]

landuse_counts["landuse_weighted_score"] = sum(
    landuse_counts[col] * weight for col, weight in land_use_weights.items()
)

# --- Join into modeling data ---
samples_df['sensor_id'] = samples_df['sensor_id'].astype(str)
landuse_counts['sensor_id'] = landuse_counts['sensor_id'].astype(str)
samples_df = samples_df.merge(landuse_counts, on='sensor_id', how='left')
samples_df.fillna(0, inplace=True)

# --- Modeling ---
print("Running regression model...")
base_features = [
    'pop_pct_change', 'pop_start', 'traffic_start',
    'log_pop_start', 'log_traffic_start', 'year_gap'
]
landuse_cols = list(set(landuse_counts.columns) - {'sensor_id'})

# --- Split into two datasets ---
df_with_landuse = samples_df[samples_df[landuse_cols].sum(axis=1) > 0].copy()
df_without_landuse = samples_df[samples_df[landuse_cols].sum(axis=1) == 0].copy()

features = base_features + ['landuse_weighted_score']

X = samples_df[features]
y = samples_df['traffic_pct_change']
# ~~~~~~~Regression~~~~~~~~~

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Feature columns used in model:", features)
print("Model coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")
coef_df = pd.DataFrame({
    'feature': features,
    'coefficient': model.coef_
})
coef_df.loc[len(coef_df.index)] = ['intercept', model.intercept_]
coef_df.to_csv("model_coefficients.csv", index=False)

# --- Save predictions ---
results_df = pd.DataFrame({
    'sensor_id': X_test.index,  # assumes X_test preserves sensor index
    'predicted_traffic_pct_change': y_pred,
    'actual_traffic_pct_change': y_test.values
})
results_df.to_csv("model_predictions.csv", index=False)

# --- Save full dataset with predictions (optional) ---
samples_df['predicted_traffic_pct_change'] = model.predict(X)
samples_df.to_csv("samples_with_predictions.csv", index=False)

print("Exported model results to CSV.")

# # ~~~~~~~Random Forest~~~~~~~~~

# Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Random Forest
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    oob_score=True
)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Random Forest Results:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"OOB Score: {rf.oob_score_:.4f}")

# Optional: Check feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop Feature Importances:")
print(importances.head(10))

# ~~~~~~~~XGB Regressor

# Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=1
)

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"XGBoost Results:\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}")

# Feature importances
importances = pd.Series(xgb.feature_importances_, index=X_train.columns)
print("\nTop Feature Importances:")
print(importances.sort_values(ascending=False).head(10))