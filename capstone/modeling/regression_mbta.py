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

# Set up the SQLAlchemy engine and session
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')
Session = sessionmaker(bind=engine)
session = Session()
class_7 = '(7) Local Road or Street'
class_5 = '(5) Major Collector'

# --- Load sensor, population, and traffic data ---
print("Loading base data...")
traffic = gpd.read_postgis(
    """
    SELECT location_id as sensor_id, town_name, functional_class, geom as geometry
    FROM general_data.traffic_nameplate
    WHERE functional_class = '(7) Local Road or Street'
    """,
    engine,
    geom_col="geometry"
)

traffic_hist = pd.read_sql("""
    SELECT tn.location_id as sensor_id, tc.start_date_time as timestamp, tc.hourly_count as volume
    FROM general_data.traffic_nameplate tn  
    LEFT JOIN general_data.traffic_counts tc ON tn.location_id = tc.location_id
    WHERE tn.functional_class = '(7) Local Road or Street'
""", engine)

pop_hist = pd.read_sql("""
    SELECT ap.year, tcc.town_name, ap.total_population as population
    FROM general_data.annual_population ap
    LEFT JOIN general_data.town_census_crosswalk tcc ON ap.zip_code = tcc.zip_code
""", engine)

# --- Load MBTA commuter rail stops ---
print("Loading MBTA stops...")
mbta_stops = gpd.read_postgis(
    """
    SELECT stop_name, town_name, geometry
    FROM general_data.commuter_rail_stops
    """,
    engine,
    geom_col="geometry"
)

# --- Geometry preparation ---
print("Preparing geometries...")
traffic = traffic[traffic.geometry.notnull() & traffic.is_valid & ~traffic.geometry.is_empty]
mbta_stops = mbta_stops[mbta_stops.geometry.notnull() & mbta_stops.is_valid & ~mbta_stops.geometry.is_empty]

# if traffic.crs is None:
#     traffic.set_crs("EPSG:4326", inplace=True)
# if mbta_stops.crs is None:
#     mbta_stops.set_crs("EPSG:4326", inplace=True)

traffic = traffic.to_crs("EPSG:26986")
mbta_stops.set_crs("EPSG:26986", inplace=True, allow_override=True)


# --- Compute distance to nearest MBTA stop ---
print("Computing distance to nearest MBTA stop...")
traffic_centroids = traffic.copy()
traffic_centroids['geometry'] = traffic_centroids.geometry.centroid

nearest = gpd.sjoin_nearest(
    traffic_centroids[['sensor_id', 'geometry']], 
    mbta_stops[['stop_name', 'town_name', 'geometry']], 
    how='left', 
    distance_col='dist_to_mbta_stop'
)
mbta_distances = nearest[['sensor_id', 'dist_to_mbta_stop']]

# --- Process traffic data to daily averages ---
print("Aggregating traffic data...")
traffic_hist['timestamp'] = pd.to_datetime(traffic_hist['timestamp'])
traffic_hist['date'] = traffic_hist['timestamp'].dt.date
traffic_hist['year'] = traffic_hist['timestamp'].dt.year
daily_avg = traffic_hist.groupby(['sensor_id', 'date'])['volume'].sum().reset_index()
daily_avg['year'] = pd.to_datetime(daily_avg['date']).dt.year
daily_avg = daily_avg.groupby(['sensor_id', 'year'])['volume'].mean().reset_index()
daily_avg.columns = ['sensor_id', 'year', 'avg_daily_volume']

# --- Filter to valid sensors ---
print("Filtering sensors with sufficient data...")
sensor_years = daily_avg.groupby('sensor_id')['year'].agg(['min', 'max', 'count'])
valid_sensors = sensor_years[(sensor_years['count'] >= 2) & ((sensor_years['max'] - sensor_years['min']) >= 1)].index
daily_avg = daily_avg[daily_avg['sensor_id'].isin(valid_sensors)]

# --- Join population data ---
pop_hist['year'] = pop_hist['year'].astype(int)
daily_avg = daily_avg.merge(traffic[['sensor_id', 'town_name']], on='sensor_id', how='left')
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

# --- Join MBTA distance ---
samples_df['sensor_id'] = samples_df['sensor_id'].astype(str)
mbta_distances['sensor_id'] = mbta_distances['sensor_id'].astype(str)
samples_df = samples_df.merge(mbta_distances, on='sensor_id', how='left')
samples_df['dist_to_mbta_stop'] = samples_df['dist_to_mbta_stop'].fillna(samples_df['dist_to_mbta_stop'].max())

# Adds interaction terms between population and year gap, and between population and MBTA distance.
samples_df['pop_pct_x_year_gap'] = samples_df['pop_pct_change'] * samples_df['year_gap']
samples_df['pop_start_x_dist_mbta'] = samples_df['pop_start'] * samples_df['dist_to_mbta_stop']

# Bins the MBTA distance into categories and one-hot encodes them for inclusion in the model.
bins = [0, 1000, 3000, 10000, np.inf]
labels = ['Very Close', 'Close', 'Far', 'Very Far']
samples_df['mbta_proximity'] = pd.cut(samples_df['dist_to_mbta_stop'], bins=bins, labels=labels)
encoded = pd.get_dummies(samples_df['mbta_proximity'], prefix='mbta_bin')
samples_df = pd.concat([samples_df, encoded], axis=1)
features = [
    'pop_pct_change', 'pop_start', 'traffic_start',
    'log_pop_start', 'log_traffic_start', 'year_gap',
    'dist_to_mbta_stop', 'pop_pct_x_year_gap', 'pop_start_x_dist_mbta'
] + list(encoded.columns)

# --- Modeling ---
print("Running regression model...")
# features = [
#     'pop_pct_change', 'pop_start', 'traffic_start',
#     'log_pop_start', 'log_traffic_start', 'year_gap', 'dist_to_mbta_stop'
# ]

X = samples_df[features]
y = samples_df['traffic_pct_change']

# Linear Regression
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

# Random Forest
print("Training Random Forest...")
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    oob_score=True
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("Random Forest Results:")
print(f"MAE: {mae_rf:.4f}, RMSE: {rmse_rf:.4f}, OOB Score: {rf.oob_score_:.4f}")
print("Top Feature Importances:\n", pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10))

# XGBoost
print("Training XGBoost...")
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
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
print("XGBoost Results:")
print(f"MAE: {mae_xgb:.4f}, RMSE: {rmse_xgb:.4f}")
print("Top Feature Importances:\n", pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False).head(10))
