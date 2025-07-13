# trip_generation_boxford.py
# Forecast trips from parcels in Boxford to other destinations in town and highway sensors

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sqlalchemy import create_engine

# --- Configuration ---
TOWN = "Boxford"
DB_URL = "postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db"

# Trip generation rates by use type (example values, can be adjusted)
TRIP_RATES = {
    "Single Family Residential": 10,
    "Multiple Houses on one parcel": 20,
    "Small Retail and Services stores (under 10,000 sq. ft.)": 30,
    "General Office Buildings": 25,
    # Add more mappings as needed
}

# --- Load Data ---
engine = create_engine(DB_URL)

parcels = gpd.read_postgis(
    f"""
    SELECT loc_id AS parcel_id, use_type, geometry
    FROM general_data.shapefiles
    WHERE town_name = '{TOWN}'
    """,
    con=engine,
    geom_col="geometry"
)

sensors = gpd.read_postgis(
    f"""
    SELECT location_id AS sensor_id, functional_class, geom AS geometry
    FROM general_data.traffic_nameplate
    WHERE town_name = '{TOWN}'
    """,
    con=engine,
    geom_col="geometry"
)

# --- Ensure CRS Compatibility ---
parcels.set_crs(epsg=26986, allow_override=True, inplace=True)
parcels = parcels.to_crs(epsg=4326)
sensors.set_crs(epsg=26986, allow_override=True, inplace=True)
sensors = sensors.to_crs(epsg=4326)

# --- Forecast Trips ---
trips = []
breakpoint()
for _, parcel in parcels.iterrows():
    rate = TRIP_RATES.get(parcel.use_type, 0)
    if rate == 0:
        continue

    origin_point = parcel.geometry.centroid

    for _, sensor in sensors.iterrows():
        dest_point = sensor.geometry
        distance_m = origin_point.distance(dest_point)

        # Simple gravity model placeholder (inverse distance weighting)
        if distance_m > 0:
            trip_count = rate / (distance_m / 1000)  # scale by km
            trips.append({
                "parcel_id": parcel.parcel_id,
                "sensor_id": sensor.sensor_id,
                "use_type": parcel.use_type,
                "distance_km": round(distance_m / 1000, 3),
                "forecasted_trips": round(trip_count, 2)
            })

trips_df = pd.DataFrame(trips)
breakpoint()
# --- Output ---
trips_df.to_csv("trip_forecast_boxford.csv", index=False)
print("Trip forecast complete. Output saved to trip_forecast_boxford.csv")
