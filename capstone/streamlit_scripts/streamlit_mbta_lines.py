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

# Assuming `session` is already defined and connected to your database

line_query = "SELECT geometry FROM general_data.commuter_rail_line WHERE geometry IS NOT NULL"
result = session.execute(text(line_query))

# Convert the result to a DataFrame
df = pd.DataFrame(result.fetchall(), columns=['geometry'])

# Function to safely load WKT geometries
def safe_wkb_load(wkb_bytes):
    try:
        # Decode WKB binary data into Shapely geometry
        return wkb.loads(wkb_bytes)
    except Exception as e:
        print(f"Error parsing WKB: {e}")

# Apply the safe WKT load function
df['geometry'] = df['geometry'].apply(safe_wkb_load)
df = df.dropna(subset=['geometry'])
gdf = gpd.GeoDataFrame(df, geometry='geometry')

#Optionally, set the CRS if you know the CRS of the geometries (e.g., EPSG:4326)
gdf.set_crs('EPSG:3857', allow_override=True, inplace=True)
gdf = gdf.to_crs('EPSG:4326')

# Extract coordinates directly for LineString geometries and flatten the list
flat_coords = [
    (lon, lat) for geom in gdf['geometry'] if geom.geom_type == 'LineString'
    for lon, lat in zip(*geom.coords.xy)
]
df_coords = pd.DataFrame(flat_coords, columns=['lon', 'lat'])


# Plot the coordinates using Plotly
st.write("Here is the map")
st.map(df_coords, size=20, color="#0044ff")
