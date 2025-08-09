import pandas as pd
import geopandas as gpd
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker
import plotly.express as px



TOWN = "Boxford"
st.set_page_config(layout="wide")


# Database setup
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')
Session = sessionmaker(bind=engine)
session = Session()

@st.cache_data
def load_data():
    # Load results (already merged model input + output)
    query = """
        SELECT sensor_id, town_name, functional_class, pop_start, pop_end, 
        traffic_start, predicted_traffic_pct_change, predicted_traffic_volume, prediction_year 
        FROM general_data.modeling_results
    """
    result = session.execute(text(query))
    results = pd.DataFrame(result.fetchall(), columns=['sensor_id', 'town_name', 'functional_class', 'pop_start', 'pop_end', 'traffic_start', 'predicted_traffic_pct_change', 'predicted_traffic_volume','prediction_year'])

    # Load parcel data
    parcels = gpd.read_postgis("""
        SELECT s.town_name, s.acres AS area_acres, s.loc_id AS pid, s.geometry, 
               s.use_type, s.transit, s.sqft, s.total_excluded
        FROM general_data.shapefiles s
        WHERE (s.total_excluded IS NULL OR s.total_excluded = 0)
    """, engine, geom_col="geometry")

    if parcels.crs is None:
        parcels.set_crs(epsg=26986, inplace=True)
    parcels = parcels.to_crs(epsg=4326)
    parcels['centroid'] = parcels.geometry.centroid
    parcels['lon'] = parcels.centroid.x
    parcels['lat'] = parcels.centroid.y

    # Traffic sensor locations
    traffic = gpd.read_postgis("""
        SELECT location_id AS sensor_id, geom AS geometry
        FROM general_data.traffic_nameplate
    """, engine, geom_col="geometry")
    traffic = traffic.to_crs(epsg=4326)
    traffic['lon'] = traffic.geometry.x
    traffic['lat'] = traffic.geometry.y

    # Rail stops
    rail_stops = gpd.read_postgis("""
        SELECT stop_name, line_id, geometry
        FROM general_data.commuter_rail_stops
    """, engine, geom_col='geometry')
    rail_stops.set_crs(epsg=26986, allow_override=True, inplace=True)
    rail_stops = rail_stops.to_crs(epsg=4326)
    rail_stops['lat'] = rail_stops.geometry.y
    rail_stops['lon'] = rail_stops.geometry.x

    # Merge results with sensor locations
    merged = results.merge(traffic[['sensor_id', 'lat', 'lon']], on="sensor_id", how="left")

    # Town geometries
    towns = gpd.read_postgis("""
        SELECT town_name, geom AS geometry
        FROM general_data.town_nameplate
    """, engine, geom_col="geometry")
    if towns.crs is None:
        towns.set_crs(epsg=26986, inplace=True)
    towns = towns.to_crs(epsg=4326)

    return parcels, merged, rail_stops, towns

parcels_df, traffic_df, rail_stops, town_geoms = load_data()

# Sidebar controls
st.sidebar.title("Filters")

# --- Town filter ---
towns_available = sorted(parcels_df["town_name"].unique())
if "town_name_selected" in st.session_state and st.session_state["town_name_selected"] in towns_available:
    default_index = towns_available.index(st.session_state["town_name_selected"])
else:
    default_index = 0

selected_town = st.sidebar.selectbox("Select Town", towns_available, index=default_index, key="town_selectbox")
st.session_state["town_name_selected"] = selected_town

# --- Functional class filter ---
functional_classes_available = ["All Classes"] + sorted(traffic_df["functional_class"].dropna().unique())
if "functional_class_selected" in st.session_state and st.session_state["functional_class_selected"] in functional_classes_available:
    func_default_index = functional_classes_available.index(st.session_state["functional_class_selected"])
else:
    func_default_index = 0

selected_functional_class = st.sidebar.selectbox("Select Functional Class", functional_classes_available, index=func_default_index,key="functional_class_selectbox")
st.session_state["functional_class_selected"] = selected_functional_class

# Subset of predictions
predictions_df = traffic_df.copy()

# Apply functional class filter
if selected_functional_class != "All Classes":
    predictions_df = predictions_df[predictions_df["functional_class"] == selected_functional_class]

# use session state to save filtered town for previous tab
towns_available = sorted(parcels_df["town_name"].unique())
if "town_name_selected" in st.session_state and st.session_state["town_name_selected"] in towns_available:
    default_index = towns_available.index(st.session_state["town_name_selected"])
else:
    default_index = 0  # Fallback to the first town if not set

# predictions_df['pct_increase'] = predictions_df['pct_increase'].fillna(0.05)
max_pct = predictions_df['predicted_traffic_pct_change'].max()
min_pct = predictions_df['predicted_traffic_pct_change'].min()
def scale_size(x): return 10 + 30 * (x - min_pct) / (max_pct - min_pct) if max_pct > min_pct else 15
predictions_df['marker_size'] = predictions_df['predicted_traffic_pct_change'].apply(scale_size)

predictions_df["hover_info"] = (
    "<b>Traffic Sensor Info</b><br>" +
    "Sensor ID: " + predictions_df["sensor_id"].astype(str) + "<br>" +
    "Functional Class" + predictions_df["functional_class"].astype(str) + "<br>" +
    "Population Change: " + predictions_df["predicted_traffic_pct_change"].map("{:.4f}%".format) + "<br>" +
    "Initial Traffic: " + predictions_df["traffic_start"].map("{:.2f}".format) + "<br>" +
    "Predicted Traffic: " + predictions_df["predicted_traffic_volume"].map("{:.2f}".format) + "<br>" + 
    "Year Prediction is Based Off: " + predictions_df["prediction_year"]
)
vmin = predictions_df['predicted_traffic_pct_change'].quantile(0.01)
vmax = predictions_df['predicted_traffic_pct_change'].quantile(0.99)
predictions_df.rename(columns={"predicted_traffic_pct_change":"Percent Change"}, inplace=True)
fig = px.scatter_mapbox(
    predictions_df,
    lat="lat",
    lon="lon",
    size="marker_size",
    color="Percent Change",
    color_continuous_scale="RdYlGn_r",
    # color_continuous_midpoint=0,
    range_color=[vmin, vmax],
    size_max=15,
    zoom=0.5,
    text='hover_info',
    title=f"Traffic Sensors Predicted Increase for Town {selected_town}"
)

fig.update_traces(hovertemplate="%{text}", mode="markers")

rail_hover_text = (
    "<b>Commuter Rail Stop Info</b><br>"+
    "Stop Name: " + rail_stops["stop_name"].astype(str) + "<br>" +
    "Line ID: " + rail_stops["line_id"].astype(str) + "<br>"
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
    hovertext=rail_hover_text,
    hoverinfo='text'
)

town_geom = town_geoms[town_geoms["town_name"] == selected_town].copy()
town_geom = town_geom.set_crs(epsg=26986, allow_override=True)

# Now reproject to lat/lon (WGS84)
town_geom = town_geom.to_crs(epsg=4326)

# Calculate centroid in lat/lon
town_centroid = town_geom.geometry.iloc[0].centroid
town_lat, town_lon = town_centroid.y, town_centroid.x

fig.update_layout(
    height=600,
    mapbox_zoom=13,
    mapbox_center={"lat": town_lat, "lon": town_lon},
    mapbox_style="open-street-map",
    margin={"r":0,"t":20,"l":0,"b":0},
)

st.title("Traffic Impact Simulator")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Predicted Traffic Impacts")

#df cleaning
display_df = predictions_df.copy()
display_df = display_df[display_df['town_name'] == st.session_state["town_name_selected"]]
display_df.rename(columns={"town_name":"Town Name","sensor_id":"Sensor ID", "functional_class":"Functional Class","predicted_traffic_pct_change":"Percent Change","predicted_traffic_volume":"Volume Change"}, inplace=True)
display_df.drop(columns=["pop_start","pop_end","traffic_start","lat","lon","marker_size", "hover_info"], inplace=True)
st.dataframe(display_df,hide_index=True)