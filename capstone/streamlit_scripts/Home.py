import geopandas as gpd
from sqlalchemy import create_engine
import streamlit as st
import plotly.express as px
import json

st.set_page_config(page_title="MBTA Communities Compliance", layout="wide")

### 1. GET DATA SECTION
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')

# Load data
gdf = gpd.read_postgis(
    sql="""
        SELECT tn.town_name, tn.current_status as compliance_status, 
               cc.classification_type, tn.geom, tn.compliance_deadline, 
               tn.total_housing, tn.min_multi_family 
        FROM general_data.town_nameplate tn 
        LEFT JOIN general_data.community_classification cc 
        ON tn.classification_id = cc.classification_id
    """,
    con=engine,
    geom_col='geom'
)

### 2. Prepare data for visualization

# Reproject to WGS84 (lat/lon)
gdf.set_crs(epsg=26986, allow_override=True, inplace=True)
gdf = gdf.to_crs(epsg=4326)

# Clean up for GeoJSON export
gdf["id"] = gdf.index.astype(str)  # Unique ID needed for plotly geojson
gdf["compliance_deadline_str"] = gdf["compliance_deadline"].dt.strftime('%Y-%m-%d')

# Convert to GeoJSON
geojson = json.loads(gdf.drop(columns="compliance_deadline").to_json())

# Define Plotly color map (must use string colors)
color_map = {
    "Compliant": "green",
    "Conditional Compliance": "lightblue",
    "Interim Compliance ": "darkblue",
    "Noncompliant": "red"
}

# Last time data was updated (manual process)
last_ingestion_date = "May 2025"

### 3. Plot data 

st.title("MBTA Communities Compliance Map")
st.write(f"This page displays the current compliance status of all towns impacted by the MBTA Communities Law as of {last_ingestion_date}")

gdf["hover_info"] = (
    "<b>Town Information</b><br>" +
    "Town Name: " + gdf["town_name"].astype(str) + "<br>" +
    "Classification Type: " + gdf["classification_type"].astype(str) + "<br>" +
    "Complaince Status: " + gdf["compliance_status"].astype(str) + "<br>" +
    "Complaince Deadline: " + gdf["compliance_deadline_str"].astype(str) + "<br>" +
    "Total Housing: " + gdf["total_housing"].map("{:.2f}".format) + "<br>" +
    "Min Multi-Family: " + gdf["min_multi_family"].map("{:.2f}".format)
)

fig = px.choropleth_mapbox(
    gdf,
    geojson=geojson,
    locations="id",
    color="compliance_status",
    color_discrete_map=color_map,
    custom_data=["hover_info"],
    mapbox_style="carto-positron",
    center={"lat": 42.3, "lon": -71.2},
    zoom=8.3,
    opacity=0.6,
)
fig.update_traces(
    hovertemplate="%{customdata[0]}<extra></extra>"
)

fig.update_layout(
    width=2000,
    height=500,
    mapbox_style="carto-positron",
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    legend_title_text="Compliance Status",
    legend=dict(
        font=dict(size=20),
        itemsizing='constant',
        tracegroupgap=10
    )
)

st.plotly_chart(fig, use_container_width=False)