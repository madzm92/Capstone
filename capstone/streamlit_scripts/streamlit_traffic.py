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
town_shapes = town_shapes.to_crs(epsg=4326)
#######

### GET MBTA STOPS

stop_points = gpd.read_postgis(
    sql="SELECT stop_name, line_id, geometry FROM general_data.commuter_rail_stops",
    con=engine,
    geom_col='geometry'
)

# Make sure it's in correct CRS
stop_points.set_crs(epsg=26986, allow_override=True, inplace=True)
stop_points = stop_points.to_crs(epsg=4326)
stop_points['latitude'] = stop_points.geometry.y
stop_points['longitude'] = stop_points.geometry.x

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
    view_state = pdk.ViewState(
        latitude=town_shapes.geometry.centroid.y.mean(),
        longitude=town_shapes.geometry.centroid.x.mean(),
        zoom=8
    )
else:
    filtered_town = town_shapes[town_shapes['town_name'] == selected_town]
    filtered_traffic = df[df['town_name'] == selected_town]
    view_state = pdk.ViewState(
        latitude=filtered_town.geometry.centroid.y.mean(),
        longitude=filtered_town.geometry.centroid.x.mean(),
        zoom=11
    )

print("shapes", filtered_town)
print("traffic points", filtered_traffic)

filtered_traffic['stop_name'] = ''
filtered_traffic['line_id'] = ''
stop_points['street_on'] = ''
stop_points['street_at'] = ''
stop_points['direction'] = ''
stop_points['latest'] = ''
# # Traffic point layer
traffic_layer = pdk.Layer(
    "ScatterplotLayer",
    filtered_traffic,
    get_position='[longitude, latitude]',
    get_fill_color='[255, 0, 0, 160]',
    get_radius=100,
    pickable=True,  # <-- Enable hover
)

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

stop_layer = pdk.Layer(
    "ScatterplotLayer",
    stop_points,
    get_position='[longitude, latitude]',
    get_fill_color='[0, 150, 255, 255]', 
    get_radius=80,
    pickable=True,
)

tooltip = {
    "html": """
    <b>Stop Name:</b> {stop_name} <br/>
    <b>Line ID:</b> {line_id} <br/>
    <b>Street On:</b> {street_on} <br/>
    <b>Street At:</b> {street_at} <br/>
    <b>Direction:</b> {direction} <br/>
    <b>Latest:</b> {latest} <br/>
    """,
    "style": {
        "backgroundColor": "steelblue",
        "color": "white",
        "fontSize": "12px"
    }
}

# Combine the map
st.pydeck_chart(pdk.Deck(
    initial_view_state=pdk.ViewState(
        latitude=filtered_town.geometry.centroid.y.mean(),
        longitude=filtered_town.geometry.centroid.x.mean(),
        zoom=11,
    ),
    layers=[polygon_layer, traffic_layer, stop_layer],
    tooltip=tooltip
))