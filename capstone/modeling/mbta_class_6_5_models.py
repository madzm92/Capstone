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

def get_distance_to_category(land_use_gdf, sensor_gdf, category_name, target_crs="EPSG:26986"):

    # Filter to group of interest
    target_land_use = land_use_gdf[land_use_gdf['grouped'] == category_name].copy()
    target_land_use = target_land_use[target_land_use.geometry.notnull() & target_land_use.is_valid]

    if target_land_use.crs.to_epsg() != 26986:
        target_land_use = target_land_use.to_crs(target_crs)
    
    # Spatial join: distance to nearest
    distance_result = gpd.sjoin_nearest(
        sensor_gdf,
        target_land_use[['geometry']],
        how='left',
        distance_col=f'dist_to_{category_name.replace(" ", "_").lower()}'
    )
    
    return distance_result[[ 'sensor_id', f'dist_to_{category_name.replace(" ", "_").lower()}']]


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
    plt.savefig(os.path.join(save_dir, f"{model_name}_residuals_vs_predicted_class_6_5.png"))
    plt.close()

    # --- 2. Histogram of Residuals ---
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residual")
    plt.title(f"{model_name} - Residual Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_residual_distribution_class_6_5.png"))
    plt.close()

# --- Load traffic sensor metadata ---
print("Loading traffic sensors...")
class_1 = "('(5) Major Collector', '(6) Minor Collector',)"
traffic = gpd.read_postgis(
    """
    SELECT location_id as sensor_id, town_name, functional_class, geom
    FROM general_data.traffic_nameplate
    where functional_class in ('(5) Major Collector', '(6) Minor Collector')
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
    where functional_class in ('(5) Major Collector', '(6) Minor Collector')
""", engine)

# --- Load population data ---
print("Loading population data...")
pop_hist = pd.read_sql("""
    SELECT ap.year, tcc.town_name, ap.total_population as population
    FROM general_data.annual_population ap
    LEFT JOIN general_data.town_census_crosswalk tcc ON ap.zip_code = tcc.zip_code
""", engine)

# --- Load land data ---
print("Loading land data...")
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

land_use = land_use[land_use.geometry.notnull() & land_use.is_valid & ~land_use.geometry.is_empty]
traffic = traffic[traffic.geometry.notnull() & traffic.is_valid & ~traffic.geometry.is_empty]

if land_use.crs is None:
    land_use.set_crs("EPSG:26986", inplace=True)

if traffic.crs is None:
    traffic.set_crs("EPSG:4326", inplace=True)

target_crs = "EPSG:26986"
land_use = land_use.to_crs(target_crs)
# traffic = traffic.to_crs(target_crs)

# sensor_buffer = traffic[['sensor_id', 'geom']].copy()
# sensor_buffer = sensor_buffer.set_geometry('geom')
# sensor_buffer['geometry'] = sensor_buffer.buffer(500)

# # --- Spatial join: parcels that intersect buffer ---
# sensor_landuse = gpd.sjoin(land_use, sensor_buffer, how='inner', predicate='intersects')
# sensor_landuse['landuse_area_sqm'] = sensor_landuse.geometry.area

# # --- Aggregate land use area by group ---
# landuse_grouped = (
#     sensor_landuse.groupby(['sensor_id', 'grouped'])['landuse_area_sqm']
#     .sum().unstack(fill_value=0)
# )

# # --- Total land use area per buffer ---
# landuse_grouped['total_landuse_area'] = landuse_grouped.sum(axis=1)

# # --- Add percentage columns ---
# for col in landuse_grouped.columns:
#     if col != 'total_landuse_area':
#         landuse_grouped[f'pct_{col}'] = landuse_grouped[col] / landuse_grouped['total_landuse_area']

# --- Reset index and merge into main dataset ---
# landuse_grouped = landuse_grouped.reset_index()
# samples_df = samples_df.merge(landuse_grouped, on='sensor_id', how='left')
# samples_df.fillna(0, inplace=True)

# sensor_landuse = gpd.sjoin_nearest(sensor_buffer, land_use, how='left', distance_col='dist_to_landuse')
# landuse_counts = (
#     sensor_landuse.groupby(['sensor_id', 'grouped']).size()
#     .unstack(fill_value=0)
#     .reset_index()
# )
######

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


#----Join land use------
distance_school = get_distance_to_category(land_use, sensor_gdf, "School/Education")
distance_commercial = get_distance_to_category(land_use, sensor_gdf, "Commercial: Office")
distance_shopping = get_distance_to_category(land_use, sensor_gdf, "Commercial: Retail")
distance_multi = get_distance_to_category(land_use, sensor_gdf, "Residential: Multi-Family")
distance_highway = get_distance_to_category(land_use, sensor_gdf, "Transportation")
distance_single = get_distance_to_category(land_use, sensor_gdf, "Residential: Single Family")
distance_religious = get_distance_to_category(land_use, sensor_gdf, "Religious")
distance_recreational = get_distance_to_category(land_use, sensor_gdf, "Recreational: Public")
distance_recreational_priv = get_distance_to_category(land_use, sensor_gdf, "Recreational: Private")
distance_agro = get_distance_to_category(land_use, sensor_gdf, "Agricultural")
distance_industry = get_distance_to_category(land_use, sensor_gdf, "Industrial")
distance_healthcare = get_distance_to_category(land_use, sensor_gdf, "Healthcare")
distance_hotel = get_distance_to_category(land_use, sensor_gdf, "Hotels/Hospitality")
distance_features = distance_school.merge(distance_shopping, on='sensor_id', how='outer') \
                                   .merge(distance_multi, on='sensor_id', how='outer') \
                                   .merge(distance_highway, on='sensor_id', how='outer') \
                                   .merge(distance_commercial, on='sensor_id', how='outer') \
                                   .merge(distance_single, on='sensor_id', how='outer') \
                                   .merge(distance_religious, on='sensor_id', how='outer') \
                                   .merge(distance_recreational, on='sensor_id', how='outer') \
                                   .merge(distance_recreational_priv, on='sensor_id', how='outer') \
                                   .merge(distance_agro, on='sensor_id', how='outer') \
                                    .merge(distance_industry, on='sensor_id', how='outer') \
                                    .merge(distance_healthcare, on='sensor_id', how='outer') \
                                    .merge(distance_hotel, on='sensor_id', how='outer')

samples_df = samples_df.merge(distance_features, on='sensor_id', how='left')
samples_df.fillna(0, inplace=True)


# --- Modeling ---
### CHANGES TO MAKE ###
# 1. Log-transform the target (traffic_pct_change)
# 2. Add interaction terms
# 3. Evaluate on back-transformed predictions
# 4. Optional: drop low-importance func_class features (already near-zero)

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

samples_df["pop_change_x_dist_to_retail"] = samples_df["pop_pct_change"] * samples_df["dist_to_commercial:_retail"]
samples_df["mbta_x_healthcare"] = samples_df["dist_to_mbta_stop"] * samples_df["dist_to_healthcare"]
samples_df["near_school"] = (samples_df["dist_to_school/education"] < 0.25).astype(int)
samples_df["near_retail"] = (samples_df["dist_to_commercial:_retail"] < 0.25).astype(int)
samples_df["retail_x_traffic"] = samples_df["dist_to_commercial:_retail"] * samples_df["traffic_start"]
samples_df["school_x_pop"] = samples_df["dist_to_school/education"] * samples_df["pop_start"]


# features = [
#     'pop_pct_change', 'pop_start', 'traffic_start',
#     'year_gap','mbta_usage', 'pop_change_x_mbta', 'pop_change_x_dist', 'mbta_x_dist',
#     'pop_change_x_dist_to_retail', 'mbta_x_healthcare', 'near_school', 'near_retail',
#     'retail_x_traffic', 'school_x_pop','dist_to_mbta_stop'
# ] + [col for col in samples_df.columns if col.startswith('func_class_')]
# # distance_cols = [col for col in samples_df.columns if col.startswith("dist_to_")]
# log_cols = [col for col in samples_df.columns if col.startswith("log_")]
# # features += distance_cols
# features += log_cols
# features.remove('log_traffic_pct_change')
# # features.remove('dist_to_commercial:_retail')
# # features.remove('dist_to_religious')

features = [
    'pop_pct_change',
    'retail_x_traffic',
    'traffic_start',
    'pop_start',
    'year_gap',
    'mbta_x_healthcare',
    'log_dist_to_commercial:_retail',
    'pop_change_x_dist',
    'log_dist_to_agricultural',
    'log_dist_to_hotels/hospitality',
    'log_dist_to_religious',
    'log_dist_to_school/education',
    'log_dist_to_healthcare',
    'school_x_pop'
]

X = samples_df[features]
y = samples_df['log_traffic_pct_change']

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Linear Regression ---
model = LinearRegression()
model.fit(X_train, y_train)
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log) - epsilon
y_true = np.expm1(y_test) - epsilon

print("\nLinear Regression:")
print(f"MAE: {mean_absolute_error(y_true, y_pred):.4f}, RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
print(f"The r2 score is {r2_score(y_true, y_pred)}")

# --- Random Forest ---
rf = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42, oob_score=True)
rf.fit(X_train, y_train)
y_pred_log_rf = rf.predict(X_test)
y_pred_rf = np.expm1(y_pred_log_rf) - epsilon

print("\nRandom Forest:")
print(f"MAE: {mean_absolute_error(y_true, y_pred_rf):.4f}, RMSE: {np.sqrt(mean_squared_error(y_true, y_pred_rf)):.4f}, OOB Score: {rf.oob_score_:.4f}")
print(f"The r2 score is {r2_score(y_true, y_pred_rf)}")
print("Top Feature Importances:")
print(pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10))

# --- XGBoost ---
xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=42)
# xgb.fit(X_train, y_train)
# y_pred_log_xgb = xgb.predict(X_test)
# y_pred_xgb = np.expm1(y_pred_log_xgb) - epsilon
xgb.fit(X_train.values, y_train.values)
y_pred_log_xgb = xgb.predict(X_test.values)
y_pred_xgb = np.expm1(y_pred_log_xgb) - epsilon

print("\nXGBoost Results:")
print(f"MAE: {mean_absolute_error(y_true, y_pred_xgb):.4f}, RMSE: {np.sqrt(mean_squared_error(y_true, y_pred_xgb)):.4f}")
print(f"The r2 score is {r2_score(y_true, y_pred_xgb)}")
print("Top Feature Importances:")
print(pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(10))

# --- Plot Residuals ---
plot_residuals(y_true, y_pred, model_name="Linear Regression")
plot_residuals(y_true, y_pred_rf, model_name="Random Forest")
plot_residuals(y_true, y_pred_xgb, model_name="XGBoost")

# from sklearn.inspection import permutation_importance

# # Assume xgb_model is your trained XGB model
# perm_result = permutation_importance(xgb, X_test, y_test, n_repeats=10, random_state=42)

# # Convert to DataFrame for sorting
# importances_df = pd.DataFrame({
#     "feature": X_test.columns,
#     "importance": perm_result.importances_mean
# }).sort_values(by="importance", ascending=False)

# print(importances_df.head(15))

# from sklearn.feature_selection import RFE
# from xgboost import XGBRegressor

# rfe_selector = RFE(estimator=XGBRegressor(n_estimators=100, random_state=42), n_features_to_select=10)
# rfe_selector = rfe_selector.fit(X_train, y_train)

# selected_features = X_train.columns[rfe_selector.support_]
# print("Selected Features:", list(selected_features))

# from sklearn.linear_model import LassoCV

# lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)

# # Select features with non-zero coefficients
# selected_features = X_train.columns[lasso.coef_ != 0]
# print("Lasso-selected features:", list(selected_features))