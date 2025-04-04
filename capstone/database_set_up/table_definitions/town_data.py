from sqlalchemy import Column, Integer, String, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class CommunityClassification(Base):

    __tablename__ = 'community_classification'
    __table_args__ = {"schema": "general_data"}

    classification_id = Column(Integer, primary_key=True)
    classification_type = Column(String)
    percent_increase = Column(String)

class TownNameplate(Base):

    __tablename__ = 'town_nameplate'
    __table_args__ = {"schema": "general_data"}

    town_name = Column(String, primary_key=True)
    county = Column(String)
    total_acres = Column(Float)
    total_sqft = Column(Float)
    classification_id = Column(Integer, ForeignKey(CommunityClassification.classification_id))
    total_housing = Column(Integer)
    min_multi_family = Column(Integer)
    min_land_area = Column(Integer) # minimum land area of community
    developable_station_area = Column(Integer) # amount of land within a half mile of the station
    percent_district_st_area = Column(Integer) # percent of district to be in station area