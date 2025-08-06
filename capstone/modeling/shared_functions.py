import geopandas as gpd
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

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
    return traffic

def load_traffic_counts(engine, functional_classes, excluded_location_ids: list = []):
    print("Loading traffic counts...")
    traffic_hist = pd.read_sql("""
        SELECT tn.location_id as sensor_id, tc.start_date_time as timestamp, tc.hourly_count as volume
        FROM general_data.traffic_nameplate tn  
        LEFT JOIN general_data.traffic_counts tc ON tn.location_id = tc.location_id
        WHERE functional_class = ANY(%(functional_classes)s)
        AND location_id != ALL(%(excluded_location_ids)s)
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