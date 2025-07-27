import geopandas as gpd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="MBTA Communities Compliance", layout="wide")

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
selected_category = st.selectbox('Select Community Category', community_categories)

towns_list = unique_towns(session, selected_category)
selected_town_name = st.selectbox('Select Town', towns_list)
df = query(session, selected_town_name)

# Convert the WKT geometries to GeoDataFrame
# Ensure geometry is properly loaded
gdf = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df['geom_wkt']))

# Set and convert CRS to EPSG:4326 for lat/lon
if gdf.crs is None:
    gdf.set_crs("EPSG:26986", inplace=True)  # Adjust based on your source CRS
gdf = gdf.to_crs(epsg=4326)

# Simplify geometries (if needed)
gdf['geometry'] = simplify_geometries(gdf)

# Calculate centroid lat/lon for each geometry
gdf['latitude'] = gdf.geometry.centroid.y
gdf['longitude'] = gdf.geometry.centroid.x

# Filter to get the selected town center
selected_town = gdf[gdf['town_name'] == selected_town_name].iloc[0]
center_lat = selected_town.latitude
center_lon = selected_town.longitude

# Group use types
gdf['use_type_group'] = gdf['use_type']
gdf.loc[gdf['use_type_group'] == 'Multiple Houses on one parcel', 'use_type_group'] = 'Multi-Unit Residence'
gdf.loc[~gdf['use_type_group'].isin(['Multi-Unit Residence', 'Single Family Residential']), 'use_type_group'] = 'Other'

# Create the Plotly choropleth
fig = px.choropleth(
    gdf,
    geojson=gdf.geometry,
    locations=gdf.index,
    color='use_type_group',
    hover_name='town_name',
    hover_data=['use_type', 'acres', 'address', 'community_category'],
)

# Update map to center on the selected town
fig.update_geos(
    visible=True,
    projection_type="albers usa",
    lakecolor="white",
    center={"lat": center_lat, "lon": center_lon},
    projection_scale=30,  # increase for more zoom
    fitbounds="locations",  # optional: fits map to geometry bounds
    scope="usa",  # optional: keep if using USA map view
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

fig.update_layout(
    width=2000,
    height=500,
    mapbox_style="carto-positron",
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    legend_title_text="Use Type Group",
    legend=dict(
        font=dict(size=20),
        itemsizing='constant',
        tracegroupgap=10
    )
)

st.write("Multi Family Housing vs Single Family Housing by Town")
fig.update_traces(marker_line_width=0.5, marker_line_color='black', selector=dict(type='choropleth'))
st.plotly_chart(fig)
