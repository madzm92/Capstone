from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from geoalchemy2 import Geometry

Base = declarative_base()

class CommuterRailLine(Base):

    __tablename__ = 'commuter_rail_line'
    __table_args__ = {"schema": "general_data"}

    # New auto-incrementing ID
    id = Column(Integer, primary_key=True, autoincrement=True)

    line_name = Column(String)
    status = Column(String)
    shape_length = Column(String)
    geometry = Column(Geometry('GEOMETRY', 4326))

class CommuterRailStops(Base):

    __tablename__ = 'commuter_rail_stops'
    __table_args__ = {"schema": "general_data"}

    # New auto-incrementing ID
    id = Column(Integer, primary_key=True, autoincrement=True)

    stop_name = Column(String)
    line_id = Column(Integer, ForeignKey(CommuterRailLine.id))
    stop_sequence = Column(Integer)
    town_name = Column(String)
    geometry = Column(Geometry('GEOMETRY', 4326))

class CommuterRailTrips(Base):

    __tablename__ = 'commuter_rail_trips'
    __table_args__ = {"schema": "general_data"}

    stop_id = Column(Integer, ForeignKey(CommuterRailStops.id), primary_key=True)
    stop_datetime = Column(DateTime, primary_key=True)
    direction_id = Column(String, primary_key=True)
    average_ons = Column(Float)
    average_offs = Column(Float)