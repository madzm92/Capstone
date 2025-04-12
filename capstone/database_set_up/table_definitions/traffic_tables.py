from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from geoalchemy2 import Geometry
from capstone.database_set_up.table_definitions.town_data import TownNameplate

Base = declarative_base()

class TrafficNameplate(Base):

    __tablename__ = 'traffic_nameplate'
    __table_args__ = {"schema": "general_data"}

    location_id = Column(String, primary_key=True)
    town_name = Column(String, ForeignKey(TownNameplate.town_name))
    street_on = Column(String)
    street_from = Column(String)
    street_to = Column(String)
    street_at = Column(String)
    direction = Column(String)
    latest = Column(DateTime)
    latitude = Column(String)
    longitude = Column(String)
    geom = Column(Geometry(geometry_type='POINT', srid=4326))  # Geo field

class TrafficCounts(Base):

    __tablename__ = 'traffic_counts'
    __table_args__ = {"schema": "general_data"}

    location_id = Column(String, ForeignKey(TrafficNameplate.location_id), primary_key=True)
    start_date_time = Column(DateTime, primary_key=True)
    first_fifteen = Column(Integer)
    second_fifteen = Column(Integer)
    third_fifteen = Column(Integer)
    fourth_fifteen = Column(Integer)
    hourly_count = Column(Integer)
    weekday = Column(String)