import pandas as pd
import re
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from shapely.geometry import Point
import geopandas as gpd

# This script converts the MassDOT traffic count locations excel download to a dataframe
# The original format of the file is not usable (stop_list.xlsx)
# The output file is traffic_locations_list.xlsx - the data is also stored in the traffic_nameplate table


#DB Set Up
DB_URI = "postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db"
engine = create_engine(DB_URI)
Session = sessionmaker(bind=engine)
session = Session()

# TRAFFIC LOCATIONS
# Load the uploaded Excel file again
file_path = "traffic_data/stop_list.xlsx"
df = pd.read_excel(file_path, sheet_name=0, header=None)

# Define regex to extract data from <td> tags
data_pattern = re.compile(r"<td[^>]*>(.*?)</td>", re.IGNORECASE)

extracted_data = []

# Iterate through column H to find HTML rows
for i in range(1, len(df)):
    html_row = df.at[i, 7]  # column H
    if isinstance(html_row, str) and "<td" in html_row:
        # Extract td values
        row_values = data_pattern.findall(html_row)
        
        if len(row_values) > 0:
            loc_id = df.at[i - 1, 8] if pd.notna(df.at[i - 1, 8]) else None  # column I, previous row
            # Create a dict with expected column names manually mapped
            record = {
                "location_id": loc_id,
                "county": row_values[0] if len(row_values) > 0 else None,
                "town_name": row_values[1] if len(row_values) > 1 else None,
                "street_on": row_values[4] if len(row_values) > 4 else None,
                "street_from": row_values[5] if len(row_values) > 5 else None,
                "street_to": row_values[6] if len(row_values) > 6 else None,
                "street_at": row_values[8] if len(row_values) > 8 else None,
                "direction": row_values[9] if len(row_values) > 9 else None,
                "latest": row_values[17] if len(row_values) > 17 else None,
                "latitude": row_values[14] if len(row_values) > 14 else None,
                "longitude": row_values[15] if len(row_values) > 15 else None,
            }
            extracted_data.append(record)

# Convert to DataFrame and show result
result_df = pd.DataFrame(extracted_data)
result_df = result_df.replace('&nbsp;', None)

# drop county, and filter df by town_name
# exclude towns that do not appear in nameplate
query = """
    SELECT distinct town_name
    FROM general_data.town_nameplate
"""
result = session.execute(text(query))
df = pd.DataFrame(result.fetchall(), columns=['town_name'])
filtered_locations_df = pd.merge(df, result_df, on='town_name')
filtered_locations_df = filtered_locations_df.drop(columns='county')

# convert lat long values to geom point value
#drop null lat/longs: only 2 rows
filtered_locations_df = filtered_locations_df.dropna(subset=['latitude', 'longitude'])
filtered_locations_df['geom'] = filtered_locations_df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
gdf = gpd.GeoDataFrame(filtered_locations_df, geometry='geom')
gdf.set_crs(epsg=4326, inplace=True)

# save to file & DB for safe keeping
# result_df.to_excel('traffic_data/filtered_traffic_locations_list.xlsx')
# gdf.to_postgis(name='traffic_nameplate', con=engine, if_exists='append', index=False, schema='general_data')

# TRAFFIC 15 MIN DATA
# Load the uploaded Excel file again
file_path = "traffic_data/traffic_data_inital_data.xlsx"
traffic_data_df = pd.read_excel(file_path, sheet_name=0, header=0)
traffic_data_df = traffic_data_df.drop(columns=['Unnamed: 0', 'Town'])

#convert column names
traffic_data_df = traffic_data_df.rename(columns={
    'Time':'time_range', '1st':'first_fifteen', '2nd':'second_fifteen','3rd':'third_fifteen','4th':'fourth_fifteen', 'Hourly count':'hourly_count','Date':'date','Weekday':'weekday', 'Loc ID':'location_id'})

#TODO: fix location ID issue. Make sure these match across files
traffic_data_df = traffic_data_df.drop_duplicates(subset=['location_id', 'time_range'])
existing_ids = gdf['location_id'].unique().tolist()
traffic_data_filtered_df = traffic_data_df[traffic_data_df['location_id'].isin(existing_ids)]

traffic_data_filtered_df.to_sql('traffic_counts', engine, schema='general_data', if_exists='append', index=False)
