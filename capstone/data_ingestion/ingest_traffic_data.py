import pandas as pd
import re
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from shapely.geometry import Point
import geopandas as gpd
from db_connection import DatabaseConnection

# This script converts the MassDOT traffic count locations excel download to a dataframe
# The original format of the file is not usable (stop_list.xlsx), and shows data within html code
# The output file is traffic_locations_list.xlsx - the data is also stored in the traffic_nameplate table

def ingest_traffic_nameplate(session):
    """Get traffic location info from the stop_list file. 
    Extract relevent values, and prepare for insertion into db.
    Insert into nameplate table"""

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
                    "functional_class": row_values[2] if len(row_values) > 4 else None,
                    "rural_urban": row_values[3] if len(row_values) > 4 else None,
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
    # exclude towns that do not appear in town_nameplate table (fk exists on town_name)
    query = """
        SELECT distinct town_name
        FROM general_data.town_nameplate
    """
    result = session.execute(text(query))
    df = pd.DataFrame(result.fetchall(), columns=['town_name'])
    filtered_locations_df = pd.merge(df, result_df, on='town_name')
    filtered_locations_df = filtered_locations_df.drop(columns='county')

    # convert lat long values to geom point value
    # drop null lat/longs: only 2 rows
    filtered_locations_df = filtered_locations_df.dropna(subset=['latitude', 'longitude'])
    filtered_locations_df['geom'] = filtered_locations_df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    traffic_nameplate_df = gpd.GeoDataFrame(filtered_locations_df, geometry='geom')
    traffic_nameplate_df.set_crs(epsg=4326, inplace=True)

    # save to file & DB for safe keeping
    result_df.to_excel('traffic_data/filtered_traffic_locations_list.xlsx')
    traffic_nameplate_df.to_postgis(name='traffic_nameplate', con=engine, if_exists='append', index=False, schema='general_data')

    return traffic_nameplate_df

def ingest_traffic_data(traffic_nameplate_df: pd.DataFrame, engine):
    """Read data from file that includes web scraped traffic instances.
    Rename columns, and validate that the location ids exist in the traffic_nameplate table (fk relationship).
    Insert data into traffic_counts table"""
    all = pd.DataFrame()
    # Load the uploaded Excel file again
    file_path = ["traffic_data/boxford_traffic_data.xlsx","traffic_data/boxford_traffic_data_2.xlsx","traffic_data/boxford_traffic_data_3.xlsx"]
    for file in file_path:
        traffic_data_df= pd.read_excel(file, sheet_name=0, header=0)
        traffic_data_df['start_date_time'] = pd.to_datetime(traffic_data_df['Date'].astype(str) + " " + traffic_data_df['Time'].str[:5] + ":00")
        traffic_data_df = traffic_data_df.drop(columns=['Unnamed: 0', 'Town', 'Date', 'Time'])
        #convert column names
        traffic_data_df = traffic_data_df.rename(columns={
            '1st':'first_fifteen', '2nd':'second_fifteen','3rd':'third_fifteen','4th':'fourth_fifteen', 'Hourly count':'hourly_count', 'Weekday':'weekday', 'Loc ID':'location_id'})

        #TODO: fix location ID issue. Make sure these match across files
        traffic_data_df = traffic_data_df.drop_duplicates(subset=['location_id', 'start_date_time'])
        existing_ids = traffic_nameplate_df['location_id'].unique().tolist()
        existing_ids_list = [str(x) for x in existing_ids]
        traffic_data_df['location_id'] = traffic_data_df['location_id'].astype(str)
        traffic_data_filtered_df = traffic_data_df[traffic_data_df['location_id'].isin(existing_ids_list)]
        all = pd.concat([all, traffic_data_filtered_df])

    all = all.drop_duplicates(keep='last')
    all.to_sql('traffic_counts', engine, schema='general_data', if_exists='append', index=False)

if __name__ == '__main__':
    db_url = "postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db"
    with DatabaseConnection(db_url) as session:
        engine = create_engine(db_url)
        traffic_nameplate_df = ingest_traffic_nameplate(session)
        ingest_traffic_data(traffic_nameplate_df, engine)
