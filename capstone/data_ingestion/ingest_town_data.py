import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text

#DB Set Up
DB_URI = "postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db"
engine = create_engine(DB_URI)
Session = sessionmaker(bind=engine)
session = Session()

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


#TODO: get county
town_df = pd.read_excel("town_data/zip_code_database.xlsx")
town_df = town_df[town_df['state'] == 'MA']
town_df = town_df.drop(columns=["zip", "type", "decommissioned", "acceptable_cities", "unacceptable_cities", "state", "timezone", "area_codes", "world_region", "country", "latitude", "longitude", "irs_estimated_population"])

town_df = town_df.drop_duplicates()
town_nameplate = pd.merge(town_nameplate, town_df, left_on='town_name', right_on='primary_city')

#get size: sum shapefiles
query = """
    SELECT town_name, sum(acres), sum(sqft)
    FROM general_data.shapefiles
    GROUP BY town_name
"""
result = session.execute(text(query))
df = pd.DataFrame(result.fetchall(), columns=['town_name', 'total_acres','total_sqft'])
town_nameplate = pd.merge(town_nameplate, df, on='town_name')

town_nameplate_order = ["town_name", "county", "total_acres", "total_sqft", "classification_id", "total_housing", "min_multi_family", "min_land_area", "developable_station_area", "percent_district_st_area"]
town_nameplate = town_nameplate[town_nameplate_order]

#insert data
community_classification_df.to_sql('community_classification', engine, schema='general_data', if_exists='append', index=False)
town_nameplate.to_sql('town_nameplate', engine, schema='general_data', if_exists='append', index=False)
