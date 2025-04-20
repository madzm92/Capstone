import pandas as pd
from sqlalchemy import create_engine, text
import geopandas as gpd
from shapely.geometry import Point
from db_connection import DatabaseConnection
import os, re

def ingest_census_crosswalk(engine):
    """Utilize the zip_code_database file from zip code file.
    Clean data and store in nameplate table"""

    zip_code_df = pd.read_excel("population_data/zip_code_database.xlsx", header=0)
    zip_code_df = zip_code_df[zip_code_df['state'] == 'MA']
    zip_code_df['zip'] = zip_code_df['zip'].astype(str)
    zip_code_df['zip'] = zip_code_df['zip'].str.zfill(width=5)
    zip_code_df = zip_code_df[['zip', 'primary_city', 'latitude', 'longitude']]
    zip_code_df = zip_code_df.rename(columns={'zip':'zip_code', 'primary_city':'town_name'})

    # exclude towns that do not appear in nameplate
    query = """
        SELECT town_name
        FROM general_data.town_nameplate
    """
    result = session.execute(text(query))
    df = pd.DataFrame(result.fetchall(), columns=['town_name'])
    zip_code_df = pd.merge(df, zip_code_df, on='town_name')

    zip_code_df['geometry'] = zip_code_df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    gdf = gpd.GeoDataFrame(zip_code_df, geometry='geometry', crs='EPSG:4326')  # WGS 84
    gdf = gdf[['zip_code', 'town_name', 'geometry']]

    # gdf.to_postgis("town_census_crosswalk", engine, if_exists="append",schema='general_data', index=False)
    return zip_code_df


def ingest_annual_data(zip_code_df, census_data_filepath, engine):
    """Ingest annual estimated population by zip code"""

    dataframes_list = []

    # iterate through all files, and store in dataframes
    for filename in os.listdir(census_data_filepath):
        if "Data" in filename and filename.endswith(".csv"):
            # Extract the year using regex
            match = re.search(r'ACSST5Y(\d{4})', filename)
            if match:
                year = match.group(1)

                # Load file
                file_path = os.path.join(census_data_filepath, filename)
                population_df = pd.read_csv(file_path, header=1)

                # Column name miss match over years
                population_df['zip_code'] = population_df['Geographic Area Name'].str.extract(r'ZCTA5 (\d{5})')
                if year in ('2011', '2012','2013','2014','2015','2016'):
                    population_df = population_df[['zip_code', 'Total!!Estimate!!Total population', 'Total!!Margin of Error!!Total population']]
                    population_df = population_df.rename(columns={'Total!!Estimate!!Total population':'total_population', 'Total!!Margin of Error!!Total population':'margin_of_error'})
                elif year in ('2017', '2018'):
                    population_df = population_df[['zip_code', 'Estimate!!Total!!Total population', 'Margin of Error!!Total MOE!!Total population']]
                    population_df = population_df.rename(columns={'Estimate!!Total!!Total population':'total_population', 'Margin of Error!!Total MOE!!Total population':'margin_of_error'})
                else:
                    population_df = population_df[['zip_code', 'Estimate!!Total!!Total population', 'Margin of Error!!Total!!Total population']]
                    population_df = population_df.rename(columns={'Estimate!!Total!!Total population':'total_population', 'Margin of Error!!Total!!Total population':'margin_of_error'})

                population_df['year'] = year
                dataframes_list.append(population_df)

    all_data_df = pd.concat(dataframes_list, ignore_index=True)
    all_data_df['margin_of_error_percent'] = all_data_df['margin_of_error']/all_data_df['total_population']
    zip_code_df = zip_code_df['zip_code']
    all_data_df = pd.merge(all_data_df, zip_code_df, on='zip_code')
    all_data_df.to_sql('annual_population', engine, schema='general_data', if_exists='append', index=False)

if __name__ == '__main__':
    db_url = "postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db"
    census_data = "population_data/census_annual"
    with DatabaseConnection(db_url) as session:
        engine = create_engine(db_url)
        zip_code_df = ingest_census_crosswalk(engine)
        ingest_annual_data(zip_code_df, census_data, engine)