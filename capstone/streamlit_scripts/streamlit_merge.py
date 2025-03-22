import geopandas as gpd
from matplotlib import pyplot as plt
from sqlalchemy import create_engine, text
from geoalchemy2 import Geometry
from sqlalchemy.orm import sessionmaker
import pandas as pd

import streamlit as st
import geopandas as gpd
import plotly.express as px
import pandas as pd



def simplify_geometries(gdf, tolerance=0.01):
    return gdf.geometry.apply(lambda geom: geom.simplify(tolerance, preserve_topology=True))

# Set up the SQLAlchemy engine and session
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')
Session = sessionmaker(bind=engine)
session = Session()

# Query the data from the table (adjust the table name and column names as needed)
query = """
    SELECT loc_id, town_name, use_type, acres, address, community_category, ST_AsText(geometry) AS geom_wkt
    FROM general_data.shapefiles
    where town_name in ('Carlisle', 'Acton')
"""
result = session.execute(text(query))

# Load the data into a DataFrame
df = pd.DataFrame(result.fetchall(), columns=['loc_id', 'town_name','use_type', 'acres','address','community_category', 'geom_wkt'])

# Close the session
session.close()

# Convert the WKT geometries to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['geom_wkt']))
acton_df = gdf[gdf['town_name'] == 'Acton']
merged_polygons = acton_df.union_all()

boundary = merged_polygons.boundary

# Now, you can plot the boundary or further process it.
# If you need to convert this boundary to a DataFrame for mapping:
filtered_gdf = gpd.GeoDataFrame(geometry=[boundary], crs=gdf.crs)

# Example for plotting the boundary:
fig = px.choropleth(
    filtered_gdf,
    geojson=filtered_gdf.geometry,
    locations=filtered_gdf.index,
)

# 5. Update map layout to use a simple choropleth style (no Mapbox)
fig.update_geos(
    visible=True,  # Set to true to enable map background
    projection_type="albers usa",  # A map projection for US, adjust as needed
    lakecolor="white",  # Set lake color to white (adjust based on preference)
    projection_scale=60,  # Adjust zoom level here (larger value zooms in)
    center={"lat": 42.4072, "lon": -71.3824},  # Center on Massachusetts (lat, lon)
    scope="usa",  # Limit scope to the USA
)
fig.update_layout(
    coloraxis_colorbar_title="Town Name",  # Add color legend title
    geo=dict(
        projection_type="albers usa",  # Ensure consistent projection for the map
        showland=True,  # Display land areas
        landcolor="lightgray",  # Color for land areas
        subunitcolor="black",  # Color for state boundaries
    ),
    margin={"r":0,"t":0,"l":0,"b":0},  # Remove margins
)

fig.update_traces(marker_line_width=0.5, marker_line_color='black', selector=dict(type='choropleth'))



# Display the map using Streamlit
st.plotly_chart(fig)