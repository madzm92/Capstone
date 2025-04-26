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
import pydeck as pdk

# Set up the SQLAlchemy engine and session
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')
Session = sessionmaker(bind=engine)
session = Session()


###GET TRAFFIC POINTS
# Query the data from the table (adjust the table name and column names as needed)
query = """
    SELECT location_id, town_name, latitude, longitude, street_at, street_on, latest, direction
    FROM general_data.traffic_nameplate
"""
result = session.execute(text(query))

# Load the data into a DataFrame
df = pd.DataFrame(result.fetchall(), columns=['location_id', 'town_name','latitude', 'longitude', 'street_at', 'street_on', 'latest', 'direction'])
# convert 
df['latitude'] = df['latitude'].astype(float)
df['longitude'] = df['longitude'].astype(float)
######

#### GET TOWN SHAPES
town_shapes = gpd.read_postgis(
    sql="SELECT town_name, geom FROM general_data.town_nameplate",
    con=engine,
    geom_col='geom'  # tell geopandas which column is the geometry
)
town_shapes.set_crs(epsg=26986, allow_override=True, inplace=True)

# Then, reproject to lat/lon
town_shapes = town_shapes.to_crs(epsg=4326)
#######

# Close the session
session.close()

st.title("🗺️ Map of Locations")

town_list = list(town_shapes['town_name'].unique())
town_list.sort()
town_list.insert(0, "All Towns")  # Add "All Towns" option at the top


selected_town = st.selectbox(
    "Select a Town",
    options=town_list,
    key="town_selectbox"
)

# Filtering
if selected_town == "All Towns":
    filtered_town = town_shapes
    filtered_traffic = df
else:
    filtered_town = town_shapes[town_shapes['town_name'] == selected_town]
    filtered_traffic = df[df['town_name'] == selected_town]

print("shapes", filtered_town)
print("traffic points", filtered_traffic)

# Traffic point layer
traffic_layer = pdk.Layer(
    "ScatterplotLayer",
    filtered_traffic,
    get_position='[longitude, latitude]',
    get_fill_color='[255, 0, 0, 160]',
    get_radius=100,
    pickable=True,  # <-- Enable hover
)

# Tooltip settings
tooltip = {
    "html": "<b>Street On:</b> {street_on} <br/>"
            "<b>Street At:</b> {street_at} <br/>"
            "<b>Direction:</b> {direction} <br/>"
            "<b>Latest Count:</b> {latest}",
    "style": {
        "backgroundColor": "steelblue",
        "color": "white"
    }
}
# Polygon layer
polygon_layer = pdk.Layer(
    "GeoJsonLayer",
    filtered_town.__geo_interface__,
    stroked=True,
    filled=True,
    get_fill_color='[0, 100, 200, 50]',
    get_line_color=[0, 0, 0],
    line_width_min_pixels=1,
)

# Combine the map
st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(
        latitude=filtered_town.geometry.centroid.y.mean(),
        longitude=filtered_town.geometry.centroid.x.mean(),
        zoom=11,
    ),
    layers=[polygon_layer, traffic_layer],
    tooltip=tooltip
))