import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from xgboost import XGBRegressor

# this model uses functional_class as a feature in addition to the mbta stop festures

# Set up the SQLAlchemy engine
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')

# --- Load sensor, population, and traffic data ---
print("Loading base data...")
traffic = gpd.read_postgis(
    """
    SELECT location_id as sensor_id, town_name, functional_class, geom
    FROM general_data.traffic_nameplate
    WHERE functional_class IS NOT NULL
    """,
    engine,
    geom_col="geom"
)

traffic_hist = pd.read_sql("""
    SELECT tn.location_id as sensor_id, tc.start_date_time as timestamp, tc.hourly_count as volume
    FROM general_data.traffic_nameplate tn  
    LEFT JOIN general_data.traffic_counts tc ON tn.location_id = tc.location_id
    WHERE tc.hourly_count IS NOT NULL
""", engine)

pop_hist = pd.read_sql("""
    SELECT ap.year, tcc.town_name, ap.total_population as population
    FROM general_data.annual_population ap
    LEFT JOIN general_data.town_census_crosswalk tcc ON ap.zip_code = tcc.zip_code
""", engine)

# --- Process traffic data to daily averages ---
traffic_hist['timestamp'] = pd.to_datetime(traffic_hist['timestamp'])
traffic_hist['date'] = traffic_hist['timestamp'].dt.date
traffic_hist['year'] = traffic_hist['timestamp'].dt.year
daily_avg = (
    traffic_hist.groupby(['sensor_id', 'date'])['volume'].sum().reset_index()
)
daily_avg['year'] = pd.to_datetime(daily_avg['date']).dt.year
daily_avg = (
    daily_avg.groupby(['sensor_id', 'year'])['volume'].mean().reset_index()
    .rename(columns={'volume': 'avg_daily_volume'})
)

# --- Filter to valid sensors ---
sensor_years = daily_avg.groupby('sensor_id')['year'].agg(['min', 'max', 'count'])
valid_sensors = sensor_years[(sensor_years['count'] >= 2) & ((sensor_years['max'] - sensor_years['min']) >= 1)].index
daily_avg = daily_avg[daily_avg['sensor_id'].isin(valid_sensors)]

# --- Join population and traffic metadata ---
pop_hist['year'] = pop_hist['year'].astype(int)
daily_avg = daily_avg.merge(
    traffic[['sensor_id', 'town_name', 'functional_class']], on='sensor_id', how='left'
)
samples = daily_avg.merge(pop_hist, on=['town_name', 'year'])

# --- Build samples ---
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
            'town_name': group.iloc[i]['town_name'],
            'functional_class': group.iloc[i]['functional_class']
        }
        samples_df.append(row)
samples_df = pd.DataFrame(samples_df)

# --- Feature engineering ---
samples_df['traffic_pct_change'] = (samples_df['traffic_end'] - samples_df['traffic_start']) / samples_df['traffic_start']
samples_df['pop_pct_change'] = (samples_df['pop_end'] - samples_df['pop_start']) / samples_df['pop_start']
samples_df['log_pop_start'] = np.log1p(samples_df['pop_start'])
samples_df['log_traffic_start'] = np.log1p(samples_df['traffic_start'])
samples_df['year_gap'] = samples_df['year_end'] - samples_df['year_start']

# --- Encode functional_class ---
samples_df = pd.get_dummies(samples_df, columns=['functional_class'], prefix='func_class')

# --- Modeling ---
base_features = [
    'pop_pct_change', 'pop_start', 'traffic_start',
    'log_pop_start', 'log_traffic_start', 'year_gap'
] + [col for col in samples_df.columns if col.startswith('func_class_')]

X = samples_df[base_features]
y = samples_df['traffic_pct_change']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # --- Linear Regression ---
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Linear Regression:")
print("MAE: {:.4f}, RMSE: {:.4f}".format(mean_absolute_error(y_test, y_pred), np.sqrt(mean_squared_error(y_test, y_pred))))

# --- Random Forest ---
rf = RandomForestRegressor(
    n_estimators=300, max_depth=10, min_samples_split=5, min_samples_leaf=2,
    random_state=42, oob_score=True
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandom Forest:")
print("MAE: {:.4f}, RMSE: {:.4f}, OOB Score: {:.4f}".format(
    mean_absolute_error(y_test, y_pred_rf),
    np.sqrt(mean_squared_error(y_test, y_pred_rf)),
    rf.oob_score_
))
print("Top Feature Importances:\n", pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10))


#~~~~~HYPER PARAM TUNING~~~~~~~
# param_grid = {
#     'n_estimators': [100, 300, 500],
#     'max_depth': [3, 4, 5, 6],
#     'learning_rate': [0.01, 0.05, 0.1],
#     'subsample': [0.6, 0.8, 1.0],
#     'colsample_bytree': [0.6, 0.8, 1.0],
#     'min_child_weight': [1, 3, 5]
# }

# from xgboost import XGBRegressor
# from sklearn.model_selection import RandomizedSearchCV

# xgb = XGBRegressor(objective='reg:squarederror', random_state=42)

# random_search = RandomizedSearchCV(
#     estimator=xgb,
#     param_distributions=param_grid,
#     n_iter=30,  # try more for better accuracy
#     scoring='neg_mean_absolute_error',
#     cv=3,
#     verbose=2,
#     random_state=42,
#     n_jobs=-1
# )

# random_search.fit(X_train, y_train)

# # Best model and results
# best_model = random_search.best_estimator_
# print("Best parameters:", random_search.best_params_)
# y_pred = best_model.predict(X_test)

# from sklearn.metrics import mean_absolute_error, mean_squared_error
# mae = mean_absolute_error(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred)

# print(f"📊 Tuned XGBoost Results:\nMAE: {mae:.4f}\nRMSE: {rmse:.4f}")

# import matplotlib.pyplot as plt
# import seaborn as sns

# importances = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# plt.figure(figsize=(10, 6))
# sns.barplot(x=importances.values[:10], y=importances.index[:10])
# plt.title("Top 10 Feature Importances (Tuned XGBoost)")
# plt.xlabel("Importance")
# plt.tight_layout()
# plt.show()

# --- XGBoost ---
xgb = XGBRegressor(
    n_estimators=100, learning_rate=0.01, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, random_state=42, min_child_weight=1
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
print("\nXGBoost:")
print("MAE: {:.4f}, RMSE: {:.4f}".format(mean_absolute_error(y_test, y_pred_xgb), np.sqrt(mean_squared_error(y_test, y_pred_xgb))))
print("Top Feature Importances:\n", pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False).head(10))
