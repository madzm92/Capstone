import pandas as pd
import streamlit as st
import geopandas as gpd
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from shapely import wkt
from sqlalchemy.orm import sessionmaker
from shapely.geometry import Point, LineString

import plotly.express as px
import matplotlib.pyplot as plt

# Database connection setup
DATABASE_URL = "postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()


from shapely import wkb
import geopandas as gpd
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sqlalchemy import text

# LineString query
line_query = "SELECT geometry FROM general_data.commuter_rail_line WHERE geometry IS NOT NULL"
result = session.execute(text(line_query))

# Convert the result to a DataFrame
df = pd.DataFrame(result.fetchall(), columns=['geometry'])

# Polygon query
query = """
    SELECT loc_id, town_name, use_type, acres, address, community_category, ST_AsText(geometry) AS geom_wkt
    FROM general_data.shapefiles
    where town_name = 'Carlisle'
"""
result = session.execute(text(query))

# Load the polygon data into a DataFrame
polygon_df = pd.DataFrame(result.fetchall(), columns=['loc_id', 'town_name', 'use_type', 'acres', 'address', 'community_category', 'geom_wkt'])

# Function to safely load WKB geometries
def safe_wkb_load(wkb_bytes):
    try:
        return wkb.loads(wkb_bytes)  # Decode WKB binary data into Shapely geometry
    except Exception as e:
        print(f"Error parsing WKB: {e}")

# Apply the safe WKB load function for LineStrings
df['geometry'] = df['geometry'].apply(safe_wkb_load)
df = df.dropna(subset=['geometry'])
gdf_lines = gpd.GeoDataFrame(df, geometry='geometry')

# Apply the safe WKT load function for Polygons (convert WKT to Shapely)
polygon_df['geometry'] = polygon_df['geom_wkt'].apply(wkt.loads)
polygon_df = polygon_df.dropna(subset=['geometry'])
gdf_polygons = gpd.GeoDataFrame(polygon_df, geometry='geometry')

# Set the CRS for both GeoDataFrames
gdf_lines.set_crs('EPSG:3857', allow_override=True, inplace=True)
gdf_polygons.set_crs('EPSG:3857', inplace=True)

if gdf_polygons.crs is None:
    print("Warning: No CRS found for shapefiles. Assigning EPSG:??? (please verify)")
    # If the polygons were stored in a different CRS, change this to match your data
    gdf_polygons.set_crs('EPSG:3857', inplace=True)  # Adjust if necessary!

# Reproject both to EPSG:4326 (lat/lon) for mapping
gdf_lines = gdf_lines.to_crs('EPSG:4326')
gdf_polygons = gdf_polygons.to_crs('EPSG:4326')

# Extract LineString coordinates
flat_coords_lines = [
    (lon, lat) for geom in gdf_lines['geometry'] if geom.geom_type == 'LineString'
    for lon, lat in zip(*geom.coords.xy)
]
df_coords_lines = pd.DataFrame(flat_coords_lines, columns=['lon', 'lat'])

# Extract Polygon coordinates (exterior only)
flat_coords_polygons = []
for geom in gdf_polygons['geometry']:
    if geom.geom_type == 'Polygon':
        flat_coords_polygons.extend(zip(*geom.exterior.coords.xy))  # Extract exterior ring coordinates
    elif geom.geom_type == 'MultiPolygon':
        for polygon in geom.geoms:  # Correctly iterate through MultiPolygon using .geoms
            flat_coords_polygons.extend(zip(*polygon.exterior.coords.xy))  # Extract exterior ring coordinates

df_coords_polygons = pd.DataFrame(flat_coords_polygons, columns=['lon', 'lat'])

# Combine the DataFrames for LineStrings and Polygons
df_coords = pd.concat([df_coords_lines, df_coords_polygons], ignore_index=True)

# Plot the coordinates using Streamlit map
st.write("Here is the map showing both LineStrings and Polygons")
st.map(df_coords)
