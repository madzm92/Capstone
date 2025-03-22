from sqlalchemy import Column, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from geoalchemy2 import Geometry

# Define the model with 'loc_id', 'wkb_geometry', and 'category'
Base = declarative_base()

class ShapeFile(Base):
    __tablename__ = 'shapefiles'
    __table_args__ = {"schema": "general_data"}

    loc_id = Column(String, primary_key=True)
    town_name = Column(String)
    community_category = Column(String)
    address = Column(String)
    owner = Column(String)
    use_codes = Column(String)
    use_type = Column(String)
    transit = Column(String) #boolean?
    acres = Column(Float)
    sqft = Column(Float)
    public_inst = Column(Float)
    non_public_exec = Column(Float)
    total_excluded = Column(Float)
    total_sensit = Column(Float)
    row = Column(Float)
    open_space = Column(Float)
    hydrology = Column(Float)
    wetlands = Column(Float)
    title_v = Column(Float)
    well_head_one = Column(Float)
    flood_shfa = Column(Float)
    farmland = Column(Float)
    surf_wat_bc = Column(Float)
    well_head_two = Column(Float)
    int_well_hea = Column(Float)
    habitat = Column(Float)
    geometry = Column(Geometry('GEOMETRY'))
    category = Column(String)  # New column for the category of the shapefile

class CommuterRailLine(Base):

    __tablename__ = 'commuter_rail_line'
    __table_args__ = {"schema": "general_data"}

    object_id = Column(String, primary_key=True)
    shape_id = Column(String)
    route_id = Column(String)
    category = Column(String)
    route_short = Column(String)
    route_long = Column(String)
    route_desc = Column(String)
    route_type = Column(String)
    route_url = Column(String)
    route_colo = Column(String)
    route_fare = Column(String)
    line_id = Column(String)
    listed_route = Column(String)
    route_ty_one = Column(String)
    created_us = Column(String)
    created_date = Column(Date)
    last_edite = Column(String)
    laste_edi_one = Column(Date)
    shape_st_len = Column(Float)
    geometry = Column(Geometry('LINESTRING', 4326))

class CommuterRailLine(Base):

    __tablename__ = 'commuter_rail_line'
    __table_args__ = {"schema": "general_data"}

    object_id = Column(String, primary_key=True)
    shape_id = Column(String)
    route_id = Column(String)
    category = Column(String)