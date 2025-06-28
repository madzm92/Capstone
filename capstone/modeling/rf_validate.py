# model_validation_boxford_rf.py
# Validate Random Forest traffic model predictions using historical data

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sklearn.metrics import mean_absolute_error, mean_squared_error as sk_mean_squared_error

# Set up the SQLAlchemy engine
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')

# --- Load Historical Traffic Data ---
traffic_hist = pd.read_sql(
    """
    SELECT tn.location_id as sensor_id, tc.start_date_time as timestamp, tc.hourly_count as volume
    FROM general_data.traffic_nameplate tn  
    LEFT JOIN general_data.traffic_counts tc ON tn.location_id = tc.location_id
    WHERE tn.town_name = 'Boxford'
    """,
    engine
)

traffic_hist['datetime'] = pd.to_datetime(traffic_hist['timestamp'])
traffic_hist = traffic_hist.dropna(subset=['datetime', 'volume'])
traffic_hist['date'] = traffic_hist['datetime'].dt.date
traffic_hist['weekday'] = traffic_hist['datetime'].dt.dayofweek  # Monday=0
traffic_hist['year'] = traffic_hist['datetime'].dt.year

# Filter to weekdays only (Monday to Friday)
traffic_hist = traffic_hist[traffic_hist['weekday'] < 5]

# --- Aggregate to Daily Totals Per Sensor ---
daily_traffic = (
    traffic_hist.groupby(['sensor_id', 'date', 'year'])['volume']
    .sum()
    .reset_index()
)

# --- Load Population History ---
pop_hist = pd.read_sql(
    """
    SELECT year, total_population
    FROM general_data.annual_population
    WHERE zip_code = '01921'
    """,
    engine
)
pop_hist['year'] = pop_hist['year'].astype(int)

# Merge population to daily traffic by year
daily_traffic = daily_traffic.merge(pop_hist, on='year', how='left')

# Drop rows with missing population values
daily_traffic = daily_traffic.dropna(subset=['total_population'])

# --- Compute average weekday volume per sensor per year ---
sensor_yearly = (
    daily_traffic.groupby(['sensor_id', 'year'])
    .agg(avg_volume=('volume', 'mean'), population=('total_population', 'mean'))
    .reset_index()
)

# added to omit covid years. remove if too many data points are removed
sensor_yearly = sensor_yearly[~sensor_yearly['year'].isin([2020, 2021, 2022])]


# Compute actual change per sensor between first and last available year
first_last = sensor_yearly.groupby('sensor_id').agg(
    first_year=('year', 'min'), last_year=('year', 'max')
).reset_index()

# Drop sensors with no year-over-year data
first_last = first_last[first_last['first_year'] != first_last['last_year']]
sensor_change = sensor_yearly[sensor_yearly['sensor_id'].isin(first_last['sensor_id'])]

# Merge to keep only first and last year for each sensor
sensor_change = sensor_change.merge(first_last, on='sensor_id')

# Pivot year to columns
pivot = sensor_change.pivot_table(
    index='sensor_id',
    columns='year',
    values='avg_volume'
).reset_index()

# Add first and last year columns back to pivot
pivot = pivot.merge(first_last, on='sensor_id')

# Safely extract values for first and last year using apply
pivot['volume_first'] = pivot.apply(lambda row: row.get(row['first_year'], np.nan), axis=1)
pivot['volume_last'] = pivot.apply(lambda row: row.get(row['last_year'], np.nan), axis=1)

# Calculate percent change
pivot['actual_change'] = ((pivot['volume_last'] - pivot['volume_first']) / pivot['volume_first']) * 100
pivot = pivot[['sensor_id', 'actual_change']].dropna(subset=['actual_change'])

# --- Load Model Predictions ---
pred_df = pd.read_csv("model_outputs_boxford_rf.csv")

# Average predicted increase across all weekdays (Monday to Friday)
pred_summary = (
    pred_df[pred_df['weekday'] < 5]
    .groupby('sensor_id')['pct_increase']
    .mean()
    .reset_index()
    .rename(columns={'pct_increase': 'predicted_pct_increase'})
)

# --- Merge Actual and Predicted ---
comparison = pivot.merge(pred_summary, on='sensor_id', how='inner')
comparison = comparison.dropna()

# --- Compute Metrics ---
mae = mean_absolute_error(comparison['actual_change'], comparison['predicted_pct_increase'])
rmse = sk_mean_squared_error(comparison['actual_change'], comparison['predicted_pct_increase'])

print("\nModel Validation Results:")
print(f"  Number of matched sensors: {len(comparison)}")
print(f"  MAE: {mae:.2f}%")
print(f"  RMSE: {rmse:.2f}%")
breakpoint()
# --- Optional: Export for inspection ---
comparison.to_csv("model_validation_comparison.csv", index=False)
print("\nDetailed comparison exported to 'model_validation_comparison.csv'")
