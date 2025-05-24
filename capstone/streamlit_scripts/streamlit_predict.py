# traffic_dashboard.py
import streamlit as st
import pandas as pd
import geopandas as gpd
import pydeck as pdk
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import matplotlib.pyplot as plt

# Set up the SQLAlchemy engine and session
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')
Session = sessionmaker(bind=engine)
session = Session()

# --- Load Data ---
@st.cache_data
def load_data():
    inputs = pd.read_csv("model_inputs_boxford.csv")
    outputs = pd.read_csv("model_outputs_boxford.csv")
    traffic = gpd.read_postgis(
        """SELECT tn.geom as geometry, tn.location_id as sensor_id 
    FROM general_data.traffic_nameplate tn 
    WHERE tn.town_name = 'Boxford'""",
        engine,
        geom_col="geometry"
    )
    traffic = traffic.to_crs("EPSG:4326")  # Convert to lat/lon for plotting
    traffic['lon'] = traffic.geometry.x
    traffic['lat'] = traffic.geometry.y
    return inputs, outputs, traffic

inputs_df, outputs_df, traffic_gdf = load_data()

# --- Sidebar ---
st.sidebar.title("Traffic Impact Viewer")
scenario_options = inputs_df["pop_increase_pct"].unique()
selected_pct = st.sidebar.selectbox("Select % Housing Increase", sorted(scenario_options))

# Join inputs and outputs by scenario_id
merged_df = outputs_df.merge(inputs_df, on="scenario_id")
filtered_df = merged_df[merged_df["pop_increase_pct"] == selected_pct]
plot_data = filtered_df.merge(traffic_gdf[["sensor_id", "lat", "lon"]], on="sensor_id")

# --- Map ---
st.title("Projected Traffic Increases in Boxford")

max_pct = plot_data["pct_increase"].max()
plot_data["radius"] = plot_data["pct_increase"].apply(lambda x: max(x * 100, 200))

# Normalize traffic increase and map to color
norm = plt.Normalize(plot_data["pct_increase"].min(), plot_data["pct_increase"].max())
cmap = plt.get_cmap("RdYlGn_r")  # Reversed: green = low, red = high
plot_data["color"] = plot_data["pct_increase"].apply(lambda x: [int(c*255) for c in cmap(norm(x))[:3]] + [180])

layer = pdk.Layer(
    "ScatterplotLayer",
    data=plot_data,
    get_position='[lon, lat]',
    get_fill_color='color',
    get_radius='radius',
    pickable=True,
    auto_highlight=True
)

view_state = pdk.ViewState(
    latitude=plot_data["lat"].mean(),
    longitude=plot_data["lon"].mean(),
    zoom=12,
    pitch=0
)

st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip={"text": "Sensor: {sensor_id}\n% Increase: {pct_increase}%"}
))

# --- Table ---
st.subheader("Details for Selected % Increase")
st.dataframe(plot_data[["sensor_id", "added_trips", "pct_increase", "distance_km"]])
