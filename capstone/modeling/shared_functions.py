import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

def get_distance_to_category(land_use_gdf, sensor_gdf, category_name, target_crs="EPSG:26986"):
    # Clean the name
    col_name = f"dist_to_{category_name.lower().replace(' ', '_').replace(':', '').replace('/', '_')}"
    
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
        distance_col=col_name
    )
    
    return distance_result[['sensor_id', col_name]]

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
    plt.savefig(os.path.join(save_dir, f"{model_name}_residuals_vs_predicted_class_2_1.png"))
    plt.close()

    # --- 2. Histogram of Residuals ---
    plt.figure(figsize=(10, 5))
    sns.histplot(residuals, bins=30, kde=True)
    plt.xlabel("Residual")
    plt.title(f"{model_name} - Residual Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{model_name}_residual_distribution_class_2_1.png"))
    plt.close()

def plot_diff(y_true, y_pred_true):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_true, y=y_pred_true, alpha=0.6, s=40)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r--', label='Ideal Fit')
    plt.xlabel('Actual Traffic Volume % Change')
    plt.ylabel('Predicted Traffic Volume % Change')
    plt.title('Predicted vs. Actual Traffic Volume (XGBoost)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def show_counts(avg_daily):
    ### SENSOR COUNTS for slide 1
    avg_daily = avg_daily[avg_daily['year'] >= 2010]
    avg_daily = avg_daily[avg_daily['year'] <= 2023]

    sensor_counts = (
        avg_daily.groupby(['sensor_id', 'year'])
        .size()
        .reset_index(name='count')
    )

    # Now compute average count per sensor, per year
    avg_counts_per_year = (
        sensor_counts.groupby('year')['count']
        .mean()
        .reset_index(name='avg_observations_per_sensor')
    )

    plt.figure(figsize=(10, 6))
    plt.plot(avg_counts_per_year['year'], avg_counts_per_year['avg_observations_per_sensor'], marker='o')

    plt.title("Average Number of Daily Observations per Sensor by Year")
    plt.xlabel("Year")
    plt.ylabel("Avg Daily Observations per Sensor")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def load_traffic_sensor_data(engine, functional_classes, excluded_location_ids):
    print("Loading traffic sensors...")

    traffic = gpd.read_postgis(
        """
        SELECT location_id as sensor_id, town_name, functional_class, geom
        FROM general_data.traffic_nameplate
        WHERE functional_class = ANY(%(functional_classes)s)
        AND location_id != ALL(%(excluded_location_ids)s)
        """,
        engine,
        geom_col="geom",
        params={"functional_classes": functional_classes,
        "excluded_location_ids": excluded_location_ids}
    )
    # Prepare geometries
    traffic = traffic[traffic.geometry.notnull() & traffic.is_valid & ~traffic.geometry.is_empty]

    if traffic.crs.to_epsg() != 26986:
        traffic = traffic.to_crs("EPSG:26986")
    return traffic

def load_traffic_counts(engine, functional_classes, excluded_location_ids: list = []):
    print("Loading traffic counts...")
    traffic_hist = pd.read_sql("""
        SELECT tn.location_id as sensor_id, tc.start_date_time as timestamp, tc.hourly_count as volume
        FROM general_data.traffic_nameplate tn  
        LEFT JOIN general_data.traffic_counts tc ON tn.location_id = tc.location_id
        WHERE functional_class = ANY(%(functional_classes)s)
        AND tn.location_id != ALL(%(excluded_location_ids)s)
    """, engine, params={"functional_classes": functional_classes, "excluded_location_ids": excluded_location_ids})

    # --- Process traffic data ---
    print("Processing traffic data...")
    traffic_hist['timestamp'] = pd.to_datetime(traffic_hist['timestamp'])
    traffic_hist['date'] = traffic_hist['timestamp'].dt.date
    traffic_hist['year'] = traffic_hist['timestamp'].dt.year

    return traffic_hist

def load_pop_data(engine):
    print("Loading population data...")
    pop_hist = pd.read_sql("""
        SELECT ap.year, tcc.town_name, ap.total_population as population
        FROM general_data.annual_population ap
        LEFT JOIN general_data.town_census_crosswalk tcc ON ap.zip_code = tcc.zip_code
    """, engine)

    return pop_hist

def load_land_use(engine):
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
    if land_use.crs is None:
        land_use.set_crs("EPSG:26986", inplace=True)

    target_crs = "EPSG:26986"
    land_use = land_use.to_crs(target_crs)

    return land_use

def get_mbta_data(engine):
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

    mbta_agg = (mbta_trips.groupby(['stop_id', 'year'])[['average_ons', 'average_offs']].mean().reset_index())
    mbta_agg['mbta_usage'] = mbta_agg['average_ons'] + mbta_agg['average_offs']

    mbta_stops = mbta_stops.merge(mbta_agg[['stop_id', 'mbta_usage']], on='stop_id', how='left')
    mbta_stops = mbta_stops[mbta_stops.geometry.notnull() & mbta_stops.is_valid & ~mbta_stops.geometry.is_empty]
    mbta_stops.set_crs("EPSG:26986", inplace=True, allow_override=True)

    return mbta_stops

def get_land_use_features(land_use, sensor_gdf, samples_df):
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
    return samples_df

def get_extra_features(samples_df):
    samples_df['traffic_pct_change'] = (samples_df['traffic_end'] - samples_df['traffic_start']) / samples_df['traffic_start']
    samples_df['pop_pct_change'] = (samples_df['pop_end'] - samples_df['pop_start']) / samples_df['pop_start']
    samples_df['log_pop_start'] = np.log1p(samples_df['pop_start'])
    samples_df['log_traffic_start'] = np.log1p(samples_df['traffic_start'])
    samples_df['year_gap'] = samples_df['year_end'] - samples_df['year_start']
    return samples_df

def evaluate(oof_true, oof_preds, xgb, X_train):
    mae = mean_absolute_error(oof_true, oof_preds)
    rmse = np.sqrt(mean_squared_error(oof_true, oof_preds))
    r2 = r2_score(oof_true, oof_preds)

    # Use raw predicted and true percentage changes directly:
    y_true = oof_true  # no expm1
    y_pred_true = oof_preds  # no expm1

    print(f"CV MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    plot_residuals(y_true, y_pred_true, model_name="XGBoost")
    print("Top Feature Importances:")
    print(pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False).head(30))

    # plot_diff(y_true, y_pred_true)

def train_model(xgb, samples_df, features):
    X = samples_df[features]
    y = samples_df['traffic_pct_change']

    # --- Set up K-Fold ---
    kf = KFold(n_splits=20, shuffle=True, random_state=42)

    # Initialize out-of-fold predictions and true values
    oof_preds = np.zeros(len(X))
    oof_true = np.zeros(len(X))

    # For tracking sensor IDs by row for error inspection later
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    samples_df = samples_df.reset_index(drop=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Training fold {fold + 1}...")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        xgb.fit(X_train, y_train)
        preds = xgb.predict(X_val)

        oof_preds[val_idx] = preds
        oof_true[val_idx] = y_val
    return oof_preds, oof_true, X_train
