# model_xgboost_boxford.py
# Train a shared XGBoost model for all sensors to predict traffic volume

import pandas as pd
import numpy as np
import geopandas as gpd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# Set up SQLAlchemy connection
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')

# --- Load traffic counts with geometry ---
traffic = gpd.read_postgis(
    """
    SELECT tn.location_id as sensor_id, tn.geom as geometry, tn.functional_class
    FROM general_data.traffic_nameplate tn
    WHERE tn.town_name = 'Boxford'
    """,
    engine,
    geom_col="geometry"
)

traffic_hist = pd.read_sql(
    """
    SELECT tc.location_id as sensor_id, tc.start_date_time as timestamp, tc.hourly_count as volume
    FROM general_data.traffic_counts tc
    INNER JOIN general_data.traffic_nameplate tn ON tc.location_id = tn.location_id
    WHERE tn.town_name = 'Boxford'
    """,
    engine
)

traffic_hist['datetime'] = pd.to_datetime(traffic_hist['timestamp'])
traffic_hist = traffic_hist.dropna(subset=['datetime', 'volume'])
traffic_hist['year'] = traffic_hist['datetime'].dt.year
traffic_hist['weekday'] = traffic_hist['datetime'].dt.dayofweek  # Monday=0
traffic_hist['date'] = traffic_hist['datetime'].dt.date

# Filter to weekdays only
traffic_hist = traffic_hist[traffic_hist['weekday'] < 5]

# Aggregate to daily total per sensor
daily_traffic = (
    traffic_hist.groupby(['sensor_id', 'date', 'year', 'weekday'])['volume']
    .sum()
    .reset_index()
)

# --- Add population ---
pop = pd.read_sql(
    """
    SELECT year, total_population
    FROM general_data.annual_population
    WHERE zip_code = '01921'
    """,
    engine
)

pop['year'] = pop['year'].astype(int)
daily_traffic = daily_traffic.merge(pop, on='year', how='left')

# --- Join sensor metadata ---
daily_traffic = daily_traffic.merge(traffic[['sensor_id', 'functional_class']], on='sensor_id', how='left')
daily_traffic = daily_traffic.dropna(subset=['total_population'])

# --- Encode categorical ---
daily_traffic['func_code'] = daily_traffic['functional_class'].astype('category').cat.codes

# --- Prepare features and target ---
X = daily_traffic[['total_population', 'weekday', 'func_code']]
y = daily_traffic['volume']

# Optional: include sensor_id as categorical feature
X['sensor_code'] = daily_traffic['sensor_id'].astype('category').cat.codes

# --- Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train XGBoost model ---
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
model.fit(X_train, y_train)
breakpoint()
# --- Evaluate ---
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred)

print("XGBoost Model Evaluation:")
print(f"  MAE: {mae:.2f}")
print(f"  RMSE: {rmse:.2f}")
