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

def query(session, town_name):
# Filter Data
    query = """
        SELECT loc_id, town_name, use_type, acres, address, community_category, ST_AsText(geometry) AS geom_wkt
        FROM general_data.shapefiles
        where town_name = :town_name
    """
    result = session.execute(text(query), {"town_name":town_name})
    df = pd.DataFrame(result.fetchall(), columns=['loc_id', 'town_name','use_type', 'acres','address','community_category', 'geom_wkt'])

    return df

def unique_towns(session, selected_category):
# Get list of Towns
    query = """
        SELECT distinct town_name
        FROM general_data.shapefiles
        where community_category = :selected_category
    """
    result = session.execute(text(query), {"selected_category":selected_category})
    rows = [row[0] for row in result]
    return sorted(rows)

def unique_community_category(session, ):
# Get list of community categories
    query = """
        SELECT distinct community_category
        FROM general_data.shapefiles
    """
    result = session.execute(text(query))
    rows = [row[0] for row in result]
    return sorted(rows)


# Set up the SQLAlchemy engine and session
engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')
Session = sessionmaker(bind=engine)
session = Session()


community_categories = unique_community_category(session)
selected_category = st.selectbox('Select a Community Category', community_categories)

towns_list = unique_towns(session, selected_category)
selected_town_name = st.selectbox('Select a Town Name', towns_list)
df = query(session, selected_town_name)

# Convert the WKT geometries to GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['geom_wkt']))

# Make sure the CRS is set correctly (e.g., EPSG:4326 for lat/long)
if gdf.crs is None:
    gdf.set_crs("EPSG:26986", inplace=True)  # Adjust if needed to your local CRS
gdf = gdf.to_crs(epsg=4326)  # Convert to WGS84 (latitude/longitude)

# Apply geometry simplification
gdf['geometry'] = simplify_geometries(gdf)
gdf['latitude'] = gdf.geometry.centroid.y
gdf['longitude'] = gdf.geometry.centroid.x

# Convert to GeoJSON format (required by Plotly)
geojson = gdf.to_crs(epsg=4326).__geo_interface__

gdf['use_type_group'] = gdf['use_type']
gdf.loc[gdf['use_type_group'] == 'Multiple Houses on one parcel', 'use_type_group'] = 'Multi-Unit Residence'
gdf.loc[~gdf['use_type_group'].isin(['Multi-Unit Residence', 'Single Family Residential']), 'use_type_group'] = 'Other'


# Create a hover feature by including other data columns (e.g., 'use_type', 'town_name')
hover_data = gdf[['town_name', 'use_type', 'acres', 'address', 'community_category']].to_dict(orient='records')

# Create the Plotly map
fig = px.choropleth(
    gdf,
    geojson=gdf.geometry,
    locations=gdf.index,
    color='use_type_group',
    hover_name='town_name',
    hover_data=['use_type', 'acres', 'address', 'community_category'],
    # title=f"Interactive Map of {selected_town_name}",
)

fig.update_geos(
    visible=True,
    projection_type="albers usa",
    lakecolor="white",
    projection_scale=20,
    center={"lat": 42.4072, "lon": -71.3824},
    scope="usa",
)
fig.update_layout(
    coloraxis_colorbar_title="Town Name", 
    geo=dict(
        projection_type="albers usa",
        showland=True,
        landcolor="lightgray",
        subunitcolor="black",
    ),
)

st.write("Multi Family Housing vs Single Family Housing by Town")
fig.update_traces(marker_line_width=0.5, marker_line_color='black', selector=dict(type='choropleth'))
st.plotly_chart(fig)
