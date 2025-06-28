import pandas as pd
import geopandas as gpd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import json
from shapely.geometry import mapping


TOWN = "Boxford"
st.set_page_config(layout="wide")


# Database setup
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')
Session = sessionmaker(bind=engine)
session = Session()

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

    # Convert parcels geometry to WGS84 lat/lon first
    parcels = parcels.to_crs(epsg=4326)

    # Now compute centroid in WGS84
    parcels['centroid'] = parcels.geometry.centroid

    # Extract lon/lat from centroid (which is now in degrees)
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

    rail_stops = gpd.read_postgis(
        sql="SELECT stop_name, line_id, geometry FROM general_data.commuter_rail_stops",
        con=engine,
        geom_col='geometry'
    )
    rail_stops.set_crs(epsg=26986, allow_override=True, inplace=True)
    rail_stops = rail_stops.to_crs(epsg=4326)
    rail_stops['lat'] = rail_stops.geometry.y
    rail_stops['lon'] = rail_stops.geometry.x

    merged = outputs.merge(inputs, on="scenario_id")
    merged = merged.merge(traffic[['sensor_id', 'lon', 'lat']], on="sensor_id", how="left")

    town_geom = gpd.read_postgis(
        f"""
        SELECT town_name, geom as geometry
        FROM general_data.town_nameplate
        WHERE town_name = '{TOWN}'
        """,
        engine,
        geom_col="geometry"
    )
    if town_geom.crs is None:
        town_geom.set_crs(epsg=26986, inplace=True)
    town_geom = town_geom.to_crs(epsg=4326)

    return parcels, merged, rail_stops, town_geom

parcels_df, traffic_df, rail_stops, town_geom = load_data()

st.title("Boxford Traffic Impact Simulator")
st.sidebar.header("User Controls")

parcel_ids = parcels_df['pid'].tolist()
selected_parcel_id = st.sidebar.selectbox("Select Parcel", parcel_ids)
selected_pct = st.sidebar.selectbox("Select % Housing Increase", sorted(traffic_df['pop_increase_pct'].unique()))
selected_rail_pct = st.sidebar.selectbox("Select % Using MBTA Commuter Rail", sorted(traffic_df['rail_pct'].unique()))

# Filter predictions for selected parcel and settings
predictions_df = traffic_df[
    (traffic_df['parcel_id'] == selected_parcel_id) &
    (traffic_df['pop_increase_pct'] == selected_pct) &
    (traffic_df['rail_pct'] == selected_rail_pct)
].copy()

if predictions_df.empty:
    st.warning("No prediction data found for the selected combination.")
else:
    import plotly.express as px

    # Prepare data for plotting
    selected = parcels_df[parcels_df['pid'] == selected_parcel_id].iloc[0]

    selected_parcel_geojson = {
        "type": "Feature",
        "geometry": mapping(selected.geometry),
        "properties": {
            "pid": selected.pid,
            "sqft": selected.sqft,
            "use_type": selected.use_type
        }
    }

    # Compute marker size scaled by pct_increase
    max_pct = predictions_df['pct_increase'].max()
    min_pct = predictions_df['pct_increase'].min()
    def scale_size(x):
        return 10 + 30 * (x - min_pct) / (max_pct - min_pct) if max_pct > min_pct else 15

    predictions_df['marker_size'] = predictions_df['pct_increase'].apply(scale_size)

    # Color scale (red for higher increase)
    fig = px.scatter_mapbox(
        predictions_df,
        lat="lat",
        lon="lon",
        size="marker_size",
        color="pct_increase",
        color_continuous_scale="RdYlGn_r",
        size_max=20,
        zoom=1,
        hover_name="sensor_id",
        hover_data={
            "pct_increase": ':.2f',
            "functional_class": True,
            "lat": False,
            "lon": False,
            "marker_size": False,
        },
        title=f"Traffic Sensors Predicted Increase for Parcel {selected_parcel_id}"
    )

    # Rail stops markers — medium size, dark blue diamond
    fig.add_scattermapbox(
        lat=rail_stops['lat'],
        lon=rail_stops['lon'],
        mode='markers+text',
        marker=dict(
            size=10,
            color='darkblue',
            opacity=0.8
        ),
        name='Rail Stops',
        hovertext=rail_stops.apply(
            lambda r: f"Rail Stop: {r['stop_name']}<br>Line: {r['line_id']}", axis=1),
        hoverinfo='text'
    )

    fig.update_layout(
        height=600,
        mapbox_zoom=13,
        mapbox_center={"lat": selected.lat, "lon": selected.lon},
        mapbox_style="open-street-map",
        margin={"r":0,"t":20,"l":0,"b":0},
    )

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center={"lat": selected.lat, "lon": selected.lon},
            zoom=13,
            layers=[
                dict(
                    sourcetype="geojson",
                    source=selected_parcel_geojson,
                    type="fill",
                    color="rgba(0, 100, 255, 0.4)",
                    below="traces"
                ),
                dict(
                    sourcetype="geojson",
                    source=selected_parcel_geojson,
                    type="line",
                    color="blue",
                    line={"width": 3},
                    below="traces"
                )
            ]
        )
    )
    fig.add_scattermapbox(
        lat=[selected.lat],
        lon=[selected.lon],
        mode='markers',
        marker=dict(size=1, color='rgba(0,0,0,0)'),  # invisible marker
        hovertemplate=(
            "<b>Parcel ID:</b> %{customdata[0]}<br>" +
            "<b>Sqft:</b> %{customdata[1]:,.0f}<br>" +
            "<b>Use Type:</b> %{customdata[2]}"
        ),
        customdata=[[selected.pid, selected.sqft, selected.use_type]],
        name='Parcel Info'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Predicted Traffic Impacts")
    st.dataframe(predictions_df[['sensor_id', 'added_trips', 'pct_increase', 'distance_km', 'functional_class']])