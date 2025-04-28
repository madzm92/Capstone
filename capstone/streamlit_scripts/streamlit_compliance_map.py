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


gdf = gpd.read_postgis(
    sql="SELECT tn.town_name, tn.current_status as compliance_status, cc.classification_type, tn.geom, tn.compliance_deadline, tn.total_housing, tn.min_multi_family FROM general_data.town_nameplate tn left join general_data.community_classification cc on tn.classification_id = cc.classification_id",
    con=engine,
    geom_col='geom'  # tell geopandas which column is the geometry
)
gdf.set_crs(epsg=26986, allow_override=True, inplace=True)
gdf = gdf.to_crs(epsg=4326)


st.title("MBTA Communities Compliance Map")

# Define color map for compliance_status
status_colors = {
    "Compliant": [0, 200, 0, 150],               # Green
    "Conditional Compliance": [173, 216, 230, 150],  # Light Blue (RGB for Light Blue)
    "Interim Compliance ": [0, 0, 139, 150],       # Dark Blue (RGB for Dark Blue)
    "Noncompliant": [255, 0, 0, 150],             # Red
}

# Build GeoJSON FeatureCollection manually
features = []
for _, row in gdf.iterrows():
    features.append({
        "type": "Feature",
        "geometry": row['geom'].__geo_interface__,
        "properties": {
            "town_name": row['town_name'],
            "classification_type": row['classification_type'],
            "compliance_status": row['compliance_status'],
            "compliance_deadline": row["compliance_deadline"].strftime('%Y-%m-%d') if pd.notnull(row['compliance_deadline']) else None,
            "total_housing": row["total_housing"],
            "min_multi_family": row['min_multi_family'],
            "fill_color": status_colors.get(row['compliance_status'], [128, 128, 128, 150]),
        }
    })

geojson = {
    "type": "FeatureCollection",
    "features": features
}

# Create Pydeck Layer
layer = pdk.Layer(
    "GeoJsonLayer",
    data=geojson,
    stroked=True,
    filled=True,
    get_fill_color="properties.fill_color",
    get_line_color=[0, 0, 0],  # Outline color
    get_line_width="properties.line_width",
    pickable=True,
    auto_highlight=True,
)

# Set initial view state
view_state = pdk.ViewState(
    latitude=42.3,
    longitude=-71.2,
    zoom=8.3,
    pitch=0
)

# Tooltip
tooltip = {
    "html": "<b>{town_name}</b>"
    "<br/>Compliance Status: {compliance_status}"
    "<br/>Classification: {classification_type}"
    "<br/>Compliance Deadline: {compliance_deadline}"
    "<br/>Total Housing: {total_housing}"
    "<br/>Minimum Multi-Family Housing: {min_multi_family}",
    "style": {"color": "white"}
}

# Show Map
st.pydeck_chart(pdk.Deck(
    layers=[layer],
    initial_view_state=view_state,
    tooltip=tooltip,
    width="100%"
),
    height=600,
    use_container_width=True
)

st.sidebar.title("Legend")

# Compliance Status Legend
st.sidebar.subheader("Compliance Status Colors:")
for status, color in status_colors.items():
    st.sidebar.markdown(
        f"<div style='display: flex; align-items: center;'>"
        f"<div style='width: 20px; height: 20px; background-color: rgba({color[0]}, {color[1]}, {color[2]}, {color[3]/255}); margin-right: 8px;'></div>"
        f"{status}</div>",
        unsafe_allow_html=True
    )
