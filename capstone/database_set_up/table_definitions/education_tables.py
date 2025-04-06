from sqlalchemy import Column, String, ForeignKey, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from capstone.database_set_up.table_definitions.town_data import TownNameplate

Base = declarative_base()


class SchoolNameplate(Base):

    __tablename__ = 'school_nameplate'
    __table_args__ = {"schema": "general_data"}

    id = Column(Integer, primary_key=True, autoincrement=True) # auto incrementing unique id
    schid = Column(String)
    school_name = Column(String)
    town_name = Column(String, ForeignKey(TownNameplate.town_name))

class AnnualEnrollment(Base):

    __tablename__ = 'annual_enrollment'
    __table_args__ = {"schema": "general_data"}

    id = Column(Integer, primary_key=True, autoincrement=True) # auto incrementing unique id
    school_id = Column(Integer, ForeignKey(SchoolNameplate.id))
    town_name = Column(String, ForeignKey(TownNameplate.town_name))
    school_year_start = Column(DateTime) 
    school_year_end = Column(DateTime) 
    total_enrolled = Column(Integer)
    grade_pk = Column(Integer)
    grade_k = Column(Integer)
    grade_one = Column(Integer)
    grade_two = Column(Integer)
    grade_three = Column(Integer)
    grade_four = Column(Integer)
    grade_five = Column(Integer)
    grade_six = Column(Integer)
    grade_seven = Column(Integer)
    grade_eight = Column(Integer)
    grade_nine = Column(Integer)
    grade_ten = Column(Integer)
    grade_eleven = Column(Integer)
    grade_twelve = Column(Integer)
