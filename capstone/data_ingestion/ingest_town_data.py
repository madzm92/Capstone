import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import os
import geopandas as gpd

#DB Set Up
DB_URI = "postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db"
engine = create_engine(DB_URI)
Session = sessionmaker(bind=engine)
session = Session()
conn = engine.raw_connection()
cursor = conn.cursor()

# community_classification data
data = {
    "classification_id": [1,2,3,4],
    "classification_type": ["Rapid Transit Community", "Commuter Rail Community", "Adjacent Community", "Adjacent Small Town"],
    "percent_increase": [25, 15, 10, 5]
}
community_classification_df = pd.DataFrame(data)

# town_nameplate data
communities_df = pd.read_excel("town_data/mbta_communities.xlsx", header=1)
communities_df = communities_df[:-5]
communities_df = communities_df.rename(columns={'Community':"town_name", "Community category":"classification_type", 
"2020 Housing Units":"total_housing", "Minimum multi-family unit capacity*":"min_multi_family","Minimum land area**":"min_land_area",
"Developable station area***":"developable_station_area", "% of district to be located in station area":"percent_district_st_area"})

#convert 
communities_df['classification_type'] = communities_df['classification_type'].str.replace('Rapid Transit', 'Rapid Transit Community')
communities_df['classification_type'] = communities_df['classification_type'].str.replace('Commuter Rail', 'Commuter Rail Community')
communities_df['classification_type'] = communities_df['classification_type'].str.replace('Adjacent community', 'Adjacent Community')
communities_df['classification_type'] = communities_df['classification_type'].str.replace('Adjacent small town', 'Adjacent Small Town')

#join data
join_df = community_classification_df[['classification_type','classification_id']]
town_nameplate = pd.merge(communities_df, join_df, on='classification_type')
town_nameplate = town_nameplate.drop(columns=['classification_type'])


#get county
town_df = pd.read_excel("population_data/zip_code_database.xlsx")
town_df = town_df[town_df['state'] == 'MA']
town_df = town_df.drop(columns=["zip", "type", "decommissioned", "acceptable_cities", "unacceptable_cities", "state", "timezone", "area_codes", "world_region", "country", "latitude", "longitude", "irs_estimated_population"])

#TODO: rename  
# Buzzards Bay -> Bourne
# Foxboro -> Foxborough
# East Freetown -> Freetown
# Middleboro -> Middleborough
# North Attleboro -> North Attleborough
# Tyngsboro -> Tyngsborough

town_df = town_df.drop_duplicates()
town_nameplate = pd.merge(town_nameplate, town_df, left_on='town_name', right_on='primary_city')


# Get correct totals for sqft & acres
# get size: sum shapefiles
full_path = os.path.join('town_data/town_whole_shapefiles/townssurvey_shp', 'TOWNSSURVEY_POLYM.shp')
main_shapefile = gpd.read_file(full_path)

main_shapefile_df = main_shapefile[['TOWN', 'AREA_ACRES', 'AREA_SQMI', 'geometry']]
main_shapefile_df = main_shapefile_df.rename(columns={'TOWN':'town_name', 'AREA_ACRES': 'total_acres', 'AREA_SQMI':'total_sqmi', 'geometry':'geom'})
main_shapefile_df['town_name'] = main_shapefile_df['town_name'].str.title()
main_shapefile_df['town_name'] = main_shapefile_df['town_name'].replace('Manchester-By-The-Sea', 'Manchester')

#update dow

town_nameplate = pd.merge(town_nameplate, main_shapefile_df, on='town_name')
town_nameplate['geom'] = town_nameplate['geom'].apply(lambda geom: geom.wkt if geom else None)

df = pd.read_csv("town_data/compliance_status.csv")
df = df[['Municipality', 'Compliance Status', 'Compliance Deadlines']]
df = df.rename(columns={"Municipality":"town_name", "Compliance Status": "current_status", "Compliance Deadlines": "compliance_deadline"})
town_nameplate = pd.merge(town_nameplate, df, on='town_name')

community_classification_df.to_sql('community_classification', engine, schema='general_data', if_exists='append', index=False)
town_nameplate.to_sql('town_nameplate', engine, schema='general_data', if_exists='append', index=False)
