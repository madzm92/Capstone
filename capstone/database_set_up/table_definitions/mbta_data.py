import geopandas as gpd
from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class MbtaNameplate(Base):

    __tablename__ = 'mbta_nameplate'
    __table_args__ = {"schema": "general_data"}

    line_name = Column(String, primary_key=True)
    stop_name = Column(String, primary_key=True)
    stop_sequence = Column(Integer)
    town_name = Column(String)
    geometry = Column(Geometry('LINESTRING', 4326))