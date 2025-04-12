from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from geoalchemy2 import Geometry
from capstone.database_set_up.table_definitions.town_data import TownNameplate

Base = declarative_base()

class TrafficNameplate(Base):

    __tablename__ = 'traffic_nameplate'
    __table_args__ = {"schema": "general_data"}

    # New auto-incrementing ID
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