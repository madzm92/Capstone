from sqlalchemy import Column, String
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class TownCensusConnect(Base):

    __tablename__ = 'town_census_connect'
    __table_args__ = {"schema": "general_data"}

    zip_code = Column(String, primary_key=True)
    census_block_id = Column(String, primary_key=True)
    town_name = Column(String, nullable=False)

class AnnualPopulation(Base):

    __tablename__ = 'annual_population'
    __table_args__ = {"schema": "general_data"}

    census_block_id = Column(String, primary_key=True)
    year = Column(String, primary_key=True)
    total_population = Column(String, nullable=False)