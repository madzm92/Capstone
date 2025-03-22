import geopandas as gpd
from sqlalchemy import create_engine, Column, Integer, String, Float, text
from sqlalchemy.ext.declarative import declarative_base
from geoalchemy2 import WKTElement
from sqlalchemy.orm import sessionmaker
import os
from geoalchemy2.types import Geometry as GeoAlchemyGeometry

from shapely.geometry import LineString

from capstone.database_set_up.table_definitions import Base, ShapeFile
from shapely import wkt

# Path to the directory containing all the shapefile folders
shapefile_base_dir = 'data_sources/Commuter_Rail_Routes/'
# file_name = '274_SOMERVILLE_detail.shp'

full_path = os.path.join(shapefile_base_dir, 'Commuter_Rail_Routes.shp')
main_shapefile = gpd.read_file(full_path)
breakpoint()
#add category type
main_shapefile['category'] = 'MBTA Lines'

# create table & insert data
DB_URI = "postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db"

# Set up SQLAlchemy and GeoAlchemy2
engine = create_engine(DB_URI)
column_replace_dict = {
    'OBJECTID':'object_id', 
    'route_shor':'route_short', 
    'listed_rou':'listed_route', 
    'route_ty_1':'route_ty_one', 
    'created_da':'created_date',
    'last_edi_1':'laste_edi_one',
    'ShapeSTLen':'shape_st_len',
    }
main_shapefile.rename(columns=column_replace_dict, inplace=True)
breakpoint()

main_shapefile['geometry'] = main_shapefile['geometry'].apply(lambda geom: geom.wkt if isinstance(geom, LineString) else geom)

dtype = {
    "geometry": GeoAlchemyGeometry("LINESTRING", srid=3857)
}

# Insert into PostGIS using Pandas
main_shapefile.to_sql(
    'commuter_rail_line',
    engine,
    schema='general_data',
    if_exists='append',
    index=False,
    dtype=dtype  # Use GeoAlchemy2 Geometry type for the geometry column
)

