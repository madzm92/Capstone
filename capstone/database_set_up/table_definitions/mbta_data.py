from sqlalchemy import Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from geoalchemy2 import Geometry

Base = declarative_base()

class MbtaNameplate(Base):

    __tablename__ = 'mbta_nameplate'
    __table_args__ = {"schema": "general_data"}

    line_name = Column(String, primary_key=True)
    stop_name = Column(String, primary_key=True)
    stop_sequence = Column(Integer)
    town_name = Column(String)
    geometry = Column(Geometry('LINESTRING', 4326))

class MbtaTrips(Base):

    __tablename__ = 'mbta_trips'
    __table_args__ = {"schema": "general_data"}

    stop_name = Column(String, primary_key=True)
    stop_datetime = Column(DateTime, primary_key=True)
    direction_id = Column(String, primary_key=True)
    average_ons = Column(Float)
    average_offs = Column(Float)