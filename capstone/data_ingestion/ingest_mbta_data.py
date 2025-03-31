import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import geopandas as gpd
import os
from geoalchemy2.types import Geometry as GeoAlchemyGeometry
from shapely.geometry import LineString, MultiLineString, Point

def mbta_lines(shapefile_base_dir):
    #get CommuterRailLine data: comes from TRAINS_RTE_TRAIN shapefile
    mbta_rte = os.path.join(shapefile_base_dir, 'TRAINS_RTE_TRAIN.shp')
    mbta_rte_df = gpd.read_file(mbta_rte)

    #INSERT mbta_rte_df INTO commuter_rail_line
    mbta_rte_df = mbta_rte_df.rename(columns={"COMM_LINE":"line_name", "COMMRAIL":"status", "SHAPE_LEN":"shape_length"})
    return mbta_rte_df

def insert_commuter_rail_line_data(mbta_rte_df):
    #1 Insert into commuter_rail_line
    mbta_rte_df['geometry'] = mbta_rte_df['geometry'].apply(lambda geom: geom.wkt if isinstance(geom, LineString) else geom)
    mbta_rte_df['geometry'] = mbta_rte_df['geometry'].apply(lambda geom: geom.wkt if isinstance(geom, MultiLineString) else geom)

    dtype = {
        "geometry": GeoAlchemyGeometry("LINESTRING", srid=3857)
    }
    mbta_rte_df.iloc[0, mbta_rte_df.columns.get_loc("line_name")] = 'Haverhill (P)'
    mbta_rte_df.iloc[4, mbta_rte_df.columns.get_loc("line_name")] = 'Middleborough/Lakeville (P)'
    mbta_rte_df.iloc[7, mbta_rte_df.columns.get_loc("line_name")] = 'Franklin/Foxboro'

    mbta_rte_df.to_sql(
        'commuter_rail_line',
        engine,
        schema='general_data',
        if_exists='append',
        index=False,
        dtype=dtype  # Use GeoAlchemy2 Geometry type for the geometry column
    )


def wrangle_trip_data():
    #get CommuterRailTrips data
    mbta_trip_df = pd.read_csv("mbta_data/MBTA_Commuter_Trips_mass_dot.csv")
    mbta_trip_df = mbta_trip_df.rename(columns={"stop_id":"stop_name", "stop_time":"stop_datetime", "route_name":"line_name", "stopsequence":"stop_sequence"})
    return mbta_trip_df

def wrangle_stop_data(shapefile_base_dir):
#Get CommuterRailStops data: comes from TRAINS_NODE shape file
    mbta_stops = os.path.join(shapefile_base_dir, 'TRAINS_NODE.shp')
    mbta_stops_df = gpd.read_file(mbta_stops)
    mbta_stops_df = mbta_stops_df.rename(columns={"STATION":"stop_name", "LINE_BRNCH":"line_name"})
    mbta_stops_df = mbta_stops_df.drop(columns=['C_RAILSTAT', 'AMTRAK', 'MAP_STA', 'STATE'])
    mbta_stops_df = mbta_stops_df.dropna()
    return mbta_stops_df

def insert_commuter_rail_stop_data(mbta_stops_new_df):
    # 2 insert data into commuter_rail_stops
    mbta_stops_new_df['geometry'] = mbta_stops_new_df['geometry'].apply(lambda geom: geom.wkt if isinstance(geom, Point) else geom)
    mbta_stops_new_df['line_name'] = mbta_stops_new_df['line_name'].replace({"South Coast Rail":"South Coast"})
    mbta_stops_new_df['stop_sequence'] = mbta_stops_new_df['stop_sequence'].fillna(0)

    #get id from 
    query = """
        SELECT id, line_name
        FROM general_data.commuter_rail_line
    """
    result = session.execute(text(query))
    line_id_df = pd.DataFrame(result.fetchall(), columns=['line_id', 'line_name'])
    mbta_stops_new_df = pd.merge(mbta_stops_new_df, line_id_df, on='line_name')
    mbta_stops_new_df.drop(columns=['line_name'], inplace=True)

    dtype = {
        "geometry": GeoAlchemyGeometry("POINT", srid=2249)
    }
    mbta_stops_new_df.to_sql(
        'commuter_rail_stops',
        engine,
        schema='general_data',
        if_exists='append',
        index=False,
        dtype=dtype  # Use GeoAlchemy2 Geometry type for the geometry column
    )

def insert_commuter_rail_trip_data(mbta_trip_df):
    #insert data into tables: commuter_rail_line, commuter_rail_stops, commuter_rail_trips

    query = """
        SELECT id, stop_name
        FROM general_data.commuter_rail_stops
    """
    result = session.execute(text(query))
    stop_id_df = pd.DataFrame(result.fetchall(), columns=['stop_id', 'stop_name'])
    mbta_trip_df = pd.merge(mbta_trip_df, stop_id_df, on='stop_name')

    #drop duplicate rows
    mbta_trip_df.drop_duplicates(inplace=True, subset=['stop_name', 'stop_datetime', 'direction_id'])
    mbta_trip_df['line_name'] = mbta_trip_df['stop_name'].replace({"Readville":"South Coast"})
    mbta_trip_df.drop(columns=['stop_name', 'line_name'], inplace=True)

    mbta_trip_df.to_sql('commuter_rail_trips', engine, schema='general_data', if_exists='append', index=False)

#DB Set Up: move to seperate file
DB_URI = "postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db"
engine = create_engine(DB_URI)
Session = sessionmaker(bind=engine)
session = Session()

###ADD TO MAIN
shapefile_base_dir = 'mbta_data/trains/'
mbta_trip_df = wrangle_trip_data()

mbta_stops_df = wrangle_stop_data(shapefile_base_dir)

mbta_rte_df = mbta_lines(shapefile_base_dir)

#DROP ALL NA: drop the words line from line name, and convert to lowercase
actual_names = mbta_rte_df['line_name'].unique()
actual_names_list = actual_names.tolist()
actual_names_list.sort()

# Not all lines in stop data exist in the lines file. These values will be dropped for now
new_names = mbta_stops_df['line_name'].unique()
new_names_list = new_names.tolist()
new_names_list.sort()

# 37 points are classiffied as lines that do not exist in the TRAINS_RTE_TRAIN file. Updated to 'Other'
# MULTIPLE, MULTIPLE (WESTERN ROUTE), NEW HAMPSHIRE MAIN, SHORE LINE, SOUTH COAST RAIL (P), STOUGHTON BRANCH
# TODO: Add back in?
names_to_update = {'South Coast Rail':'South Coast', 'CAPE COD MAIN LINE':'CapeFLYER', 'FAIRMOUNT LINE':'Fairmount', 'FITCHBURG LINE':'Fitchburg', 'FOXBORO (SPECIAL EVENTS ONLY)':'Foxboro', 'FRAMINGHAM/WORCESTER LINE':'Framingham/Worcester', 'FRANKLIN LINE':'Franklin', 'FRANKLIN LINE(P)': 'Franklin', 'GREENBUSH LINE': 'Greenbush', 'HAVERHILL LINE': 'Haverhill', 'KINGSTON LINE':'Kingston', 'LOWELL LINE':'Lowell', 'MIDDLEBOROUGH MAIN': 'Middleborough/Lakeville', 'MIDDLEBOROUGH/LAKEVILLE LINE': 'Middleborough/Lakeville', 'MULTIPLE': 'Other', 'MULTIPLE (WESTERN ROUTE)': 'Other', 'NEEDHAM LINE':'Needham', 'NEW HAMPSHIRE MAIN':'Other', 'NEWBURYPORT LINE': 'Newburyport/Rockport', 'NEWBURYPORT/ROCKPORT LINE': 'Newburyport/Rockport', 'PROVIDENCE/STOUGHTON LINE': 'Providence/Stoughton', 'ROCKPORT LINE': 'Newburyport/Rockport', 'SHORE LINE': 'Other', 'SOUTH COAST RAIL (P)': 'Other', 'STOUGHTON BRANCH': 'Other'}
mbta_stops_df['line_name'] = mbta_stops_df['line_name'].replace(names_to_update)

# update stop names between mbta_stops_df & mbta_trip_df
stop_shapefile_names = mbta_stops_df['stop_name'].unique()
stop_shapefile_names_list = stop_shapefile_names.tolist()
stop_shapefile_names_list.sort()
updated_stop_shapefile_names_dict = {}
updated_stop_shapefile_names_list = []
for word in stop_shapefile_names_list:
    new_word = word.title()
    updated_stop_shapefile_names_dict.update({word:new_word})
    updated_stop_shapefile_names_list.append(new_word)

#rename values in df to new values
mbta_stops_df['stop_name'] = mbta_stops_df['stop_name'].replace(updated_stop_shapefile_names_dict)


# stop names for trips data
stop_trips_names = mbta_trip_df['stop_name'].unique()
stop_trips_names_list = stop_trips_names.tolist()
stop_trips_names_list.sort()
updated_stop_trips_names_dict = {}
updated_stop_trips_names_list = []
for word in stop_trips_names_list:
    new_word = word.replace(" / ", "/")
    new_word = new_word.replace(" /", "/")
    new_word = new_word.replace("/ ", "/")
    updated_stop_trips_names_dict.update({word:new_word})
    updated_stop_trips_names_list.append(new_word)

#rename values in df to new values
mbta_trip_df['stop_name'] = mbta_trip_df['stop_name'].replace(updated_stop_trips_names_dict)

#Plimptonville, Plymouth do not exist: closed in 2021
# update stop_trips_names df to include original names for 18 mismatched stops
rename_dict = {'South Coast Rail':'South Coast', 'Dedham Corp Center':'Dedham Corp. Center','Four Corners / Geneva': 'Four Corners/Geneva Ave','Franklin':'Franklin/Dean College','JFK/UMASS':'Jfk/Umass','Littleton / Rte 495': 'Littleton/Route 495','Melrose Cedar Park':'Melrose/Cedar Park', 'River Works/GE Employees Only':'River Works', 'TF Green Airport':'Tf Green Airport', 'Yawkey':'Lansdowne'}

#rename values in df to new values
mbta_trip_df['stop_name'] = mbta_trip_df['stop_name'].replace(rename_dict)

#add stop sequence
stop_sequence_df = mbta_trip_df[['stop_name', 'stop_sequence']]
stop_sequence_df.drop_duplicates(inplace=True)
mbta_stops_df = pd.merge(mbta_stops_df, stop_sequence_df, on='stop_name', how="left")

# Manually updated file
mbta_all_df = pd.read_csv("mbta_data/mbta_lines_stops_towns.csv")

mbta_stops_new_df = mbta_all_df[['stop_name', 'line_name', 'stop_sequence', 'town_name', 'geometry']]

# wrangle commuter_rail_trips data from mbta_trip_df
# get stop_name, stop_datetime, direction_id, average_ons, average_offs
mbta_trip_df = mbta_trip_df[['stop_name', 'stop_datetime', 'direction_id', 'average_ons', 'average_offs']]

#TODO: fix date issues!!!!!!
mbta_trip_df['stop_datetime'] = pd.to_datetime(mbta_trip_df['stop_datetime'], format='%Y/%m/%d %H:%M:%S%z')


insert_commuter_rail_line_data(mbta_rte_df)

insert_commuter_rail_stop_data(mbta_stops_new_df)

insert_commuter_rail_trip_data(mbta_trip_df)
