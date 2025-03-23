import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

# use file MBTA_Commuter_Trips_mass_dot

#DB Set Up
DB_URI = "postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db"
engine = create_engine(DB_URI)
Session = sessionmaker(bind=engine)
session = Session()

mbta_df = pd.read_csv("mbta_data/MBTA_Commuter_Trips_mass_dot.csv")
mbta_df = mbta_df.rename(columns={"stop_id":"stop_name", "stop_time":"stop_datetime", "route_name":"line_name", "stopsequence":"stop_sequence"})


mbta_nameplate_df = mbta_df[["line_name", "stop_name", "stop_sequence"]]
#TODO: get town_name & geometry from shapefiles

mbta_trips_df = mbta_df[["stop_name", "stop_datetime", "direction_id", "average_ons", "average_offs"]]
mbta_trips_df = mbta_trips_df.fillna(0)

breakpoint()

