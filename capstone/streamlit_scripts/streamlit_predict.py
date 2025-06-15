# traffic_model_boxford_rf.py
# Display traffic impact predictions for Boxford using precomputed results

import pandas as pd
import geopandas as gpd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pydeck as pdk
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from shapely.geometry import mapping

# Set up the SQLAlchemy engine and session
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')
Session = sessionmaker(bind=engine)
session = Session()

TOWN = "Boxford"

@st.cache_data
def load_data():
    inputs = pd.read_csv("model_inputs_boxford_rf.csv")
    outputs = pd.read_csv("model_outputs_boxford_rf.csv")

    parcels = gpd.read_postgis(
        f"""
        SELECT s.town_name, s.acres as area_acres, s.loc_id as pid, s.geometry, 
               s.use_type, s.transit, s.sqft, s.total_excluded
        FROM general_data.shapefiles s
        WHERE s.town_name = '{TOWN}' AND (s.total_excluded IS NULL OR s.total_excluded = 0)
        """,
        engine,
        geom_col="geometry"
    )

    if parcels.crs is None:
        parcels.set_crs(epsg=26986, inplace=True)

    parcels['centroid'] = parcels.geometry.centroid
    parcels = parcels.to_crs(epsg=4326)
    parcels['lon'] = parcels.centroid.x
    parcels['lat'] = parcels.centroid.y

    traffic = gpd.read_postgis(
        f"""
        SELECT tn.geom as geometry, tn.location_id as sensor_id, tn.functional_class
        FROM general_data.traffic_nameplate tn 
        WHERE tn.town_name = '{TOWN}'
        """,
        engine,
        geom_col="geometry"
    )

    traffic = traffic.to_crs(epsg=4326)
    traffic['lon'] = traffic.geometry.x
    traffic['lat'] = traffic.geometry.y

    merged = outputs.merge(inputs, on="scenario_id")
    merged = merged.merge(traffic[['sensor_id', 'lon', 'lat']], on="sensor_id", how="left")

    return parcels, merged

parcels_df, traffic_df = load_data()

st.title("Boxford Traffic Impact Simulator")
st.sidebar.header("User Controls")

parcel_ids = parcels_df['pid'].tolist()
selected_parcel_id = st.sidebar.selectbox("Select Parcel", parcel_ids)
selected_pct = st.sidebar.selectbox("Select % Housing Increase", sorted(traffic_df['pop_increase_pct'].unique()))

# Filter predictions for selected parcel and percent
predictions_df = traffic_df[
    (traffic_df['parcel_id'] == selected_parcel_id) &
    (traffic_df['pop_increase_pct'] == selected_pct)
].copy()

st.subheader("Predicted Traffic Impacts")
st.dataframe(predictions_df[['sensor_id', 'added_trips', 'pct_increase', 'distance_km', 'functional_class']])

# --- Map Visualization ---
max_pct = predictions_df["pct_increase"].max()
predictions_df["radius"] = predictions_df["pct_increase"].apply(lambda x: max(x * 40, 100))

norm = plt.Normalize(predictions_df["pct_increase"].min(), max_pct)
cmap = plt.get_cmap("RdYlGn_r")
predictions_df["color"] = predictions_df["pct_increase"].apply(lambda x: [int(c*255) for c in cmap(norm(x))[:3]] + [180])

sensor_layer = pdk.Layer(
    "ScatterplotLayer",
    data=predictions_df,
    get_position='[lon, lat]',
    get_fill_color='color',
    get_radius='radius',
    pickable=True,
    auto_highlight=True
)

selected_parcel = parcels_df[parcels_df['pid'] == selected_parcel_id]

# Create labeled GeoJSON for parcel with metadata in properties
parcel_feature = {
    "type": "Feature",
    "geometry": mapping(selected_parcel.iloc[0].geometry),
    "properties": {
        "use_type": selected_parcel.iloc[0].use_type,
        "sqft": selected_parcel.iloc[0].sqft,
        "transit": selected_parcel.iloc[0].transit
    }
}
parcel_geojson = {
    "type": "FeatureCollection",
    "features": [parcel_feature]
}

parcel_layer = pdk.Layer(
    "GeoJsonLayer",
    data=parcel_geojson,
    get_fill_color='[0, 100, 255, 80]',
    get_line_color='[0, 100, 255]',
    line_width_min_pixels=1,
    pickable=True,
    get_line_width=4,
)

view_state = pdk.ViewState(
    latitude=selected_parcel['lat'].values[0],
    longitude=selected_parcel['lon'].values[0],
    zoom=13,
    pitch=0
)

st.pydeck_chart(pdk.Deck(
    layers=[sensor_layer, parcel_layer],
    initial_view_state=view_state,
    tooltip={
        "html": """
        <b>Sensor:</b> {sensor_id}<br/>
        <b>% Increase:</b> {pct_increase}%<br/>
        <b>Use Type:</b> {use_type}<br/>
        <b>Sqft:</b> {sqft}<br/>
        <b>Transit:</b> {transit}
        """,
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
))
