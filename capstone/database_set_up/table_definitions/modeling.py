from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from geoalchemy2 import Geometry

Base = declarative_base()

class ModelingResultd(Base):

    __tablename__ = 'modeling_results'
    __table_args__ = {"schema": "general_data"}

    sensor_id = Column(String, primary_key=True)
    town_name = Column(String)
    functional_class = Column(String)
    traffic_year = Column(Integer)
    pop_start = Column(Float)
    pop_end = Column(Float)
    traffic_start = Column(Float)
    predicted_traffic_pct_change = Column(Float)
    predicted_traffic_volume = Column(Float)
