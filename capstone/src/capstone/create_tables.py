import geopandas as gpd
from sqlalchemy import create_engine, Column, Integer, String, Float, text
from sqlalchemy.ext.declarative import declarative_base
from geoalchemy2 import WKTElement
from sqlalchemy.orm import sessionmaker
import os
from table import Base, ShapeFile

# create table & insert data
DB_URI = "postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db"

# Set up SQLAlchemy and GeoAlchemy2
engine = create_engine(DB_URI)

with engine.connect() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS general_data;"))  # Create schema
    conn.commit()

Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)