from sqlalchemy import Column, String, ForeignKey, Integer, Float
from sqlalchemy.ext.declarative import declarative_base
from geoalchemy2 import Geometry
from capstone.database_set_up.table_definitions.town_data import TownNameplate


Base = declarative_base()


class TownCensusCrosswalk(Base):

    __tablename__ = 'town_census_crosswalk'
    __table_args__ = {"schema": "general_data"}

    zip_code = Column(String, primary_key=True)
    town_name = Column(String, ForeignKey(TownNameplate.town_name))
    geometry = Column(Geometry('GEOMETRY', 4326))

class AnnualPopulation(Base):

    __tablename__ = 'annual_population'
    __table_args__ = {"schema": "general_data"}

    zip_code = Column(String, ForeignKey(TownCensusCrosswalk.zip_code), primary_key=True)
    year = Column(String, primary_key=True)
    total_population = Column(Integer)
    margin_of_error = Column(Integer)
    margin_of_error_percent = Column(Float)