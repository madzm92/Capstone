import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

#DB Set Up
DB_URI = "postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db"
engine = create_engine(DB_URI)
Session = sessionmaker(bind=engine)
session = Session()

education_df = pd.read_csv('education_data/education_enrollment_by_year_schools_2005-2024.csv')

# exclude towns that do not appear in nameplate
query = """
    SELECT town_name
    FROM general_data.town_nameplate
"""
result = session.execute(text(query))
df = pd.DataFrame(result.fetchall(), columns=['town_name'])
education_df = pd.merge(df, education_df, right_on='municipal', left_on='town_name')

# wrangle nameplate data: school_name, school_id, town_name
education_df[['town_name', 'school_name']] = education_df['name'].str.split(' - ',n=1, expand=True)
education_df['school_name'] = education_df['school_name'].fillna(education_df['name'])
education_nameplate_df = education_df.copy()
education_nameplate_df = education_nameplate_df[['schid','school_name', 'municipal']]
education_nameplate_df = education_nameplate_df.rename(columns={'municipal':'town_name'})
education_nameplate_df = education_nameplate_df.drop_duplicates()

#wrangle annual_enrollment
education_enrollment_df = education_df.copy()

education_enrollment_df['school_year_start'] = education_enrollment_df['schoolyear'].str.slice(0, 4)
education_enrollment_df['school_year_end'] = '20' + education_enrollment_df['schoolyear'].str.slice(5, 7)

education_enrollment_df = education_enrollment_df[['municipal','school_name', 'school_year_start', 'school_year_end', 'enrolled',
       'grade_pk', 'grade_k', 'grade_1', 'grade_2', 'grade_3', 'grade_4',
       'grade_5', 'grade_6', 'grade_7', 'grade_8', 'grade_9', 'grade_10',
       'grade_11', 'grade_12']]

education_enrollment_df = education_enrollment_df.rename(columns={
    'municipal':'town_name',
    'enrolled':'total_enrolled',
    'grade_1':'grade_one',
    'grade_2':'grade_two',
    'grade_3':'grade_three',
    'grade_4':'grade_four',
    'grade_5':'grade_five',
    'grade_6':'grade_six',
    'grade_7':'grade_seven',
    'grade_8':'grade_eight',
    'grade_9':'grade_nine',
    'grade_10':'grade_ten',
    'grade_11':'grade_eleven',
    'grade_12':'grade_twelve',
    })

education_enrollment_df['school_year_start'] = pd.to_datetime(education_enrollment_df['school_year_start'], format='%Y')
education_enrollment_df['school_year_end'] = pd.to_datetime(education_enrollment_df['school_year_end'], format='%Y')

# get the school id
education_nameplate_id_df = education_nameplate_df.copy()
education_nameplate_id_df['id'] = pd.RangeIndex(start=1, stop=len(education_nameplate_id_df) + 1)
education_nameplate_id_df = education_nameplate_id_df[['school_name', 'id']]
education_enrollment_df = pd.merge(education_enrollment_df, education_nameplate_id_df, on='school_name')
education_enrollment_df = education_enrollment_df.rename(columns={'id':'school_id'})
education_enrollment_df = education_enrollment_df[['school_id','town_name', 'school_year_start','school_year_end', 
                                                   'total_enrolled', 'grade_pk', 'grade_k', 'grade_one', 'grade_two',
       'grade_three', 'grade_four', 'grade_five', 'grade_six', 'grade_seven',
       'grade_eight', 'grade_nine', 'grade_ten', 'grade_eleven',
       'grade_twelve']]

#insert data
education_nameplate_df.to_sql('school_nameplate', engine, schema='general_data', if_exists='append', index=False)
education_enrollment_df.to_sql('annual_enrollment', engine, schema='general_data', if_exists='append', index=False)
