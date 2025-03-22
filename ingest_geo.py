import os
import geopandas as gpd
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from geoalchemy2 import Geometry
from sqlalchemy.orm import sessionmaker
from shapely.geometry import Point, Polygon
from sqlalchemy.exc import SQLAlchemyError

# Define your PostgreSQL connection string
DB_URI = "postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db"

# Set up SQLAlchemy and GeoAlchemy2
Base = declarative_base()
engine = create_engine(DB_URI)

# Define the model with 'loc_id', 'wkb_geometry', and 'category'
class ShapeFile(Base):
    __tablename__ = 'shape_files'
    loc_id = Column(Integer, primary_key=True)
    wkb_geometry = Column(Geometry('MULTIPOLYGON'))
    category = Column(String)  # New column for the category of the shapefile

# Create a session to interact with the database
Session = sessionmaker(bind=engine)
session = Session()

# Path to the directory containing all the shapefile folders
shapefile_base_dir = 'shape_file_folders/51_CARLISLE_basic'

# Function to determine the category based on folder name
def get_category_from_folder(folder_name):
    folder_name = folder_name.lower()
    if folder_name.startswith('excluded'):
        return 'excluded'
    elif folder_name.startswith('sensitive'):
        return 'sensitive'
    elif folder_name.startswith('transit station'):
        return 'transit station'
    elif folder_name.startswith('density denominatory'):
        return 'Density Denominatory'
    elif folder_name.startswith('actual shapefiles'):
        return 'actual_shapefiles'
    else:
        return 'other'  # Default category if no match is found

# Function to find the ID column dynamically
def find_id_column(columns):
    for col in columns:
        if "ID" in col.upper():
            return col
    return None  # Return None if no matching column is found

# Iterate through all folders in the base directory
for root, dirs, files in os.walk(shapefile_base_dir):
    for file in files:
        if file.endswith('.shp'):
            shapefile_path = os.path.join(root, file)

            # Get the category from the folder name
            category = get_category_from_folder(os.path.basename(root))

            # Read the shapefile using GeoPandas
            gdf = gpd.read_file(shapefile_path)

            # Dynamically find the ID column
            id_column = find_id_column(gdf.columns)

            if id_column is None:
                print(f"Warning: No ID column found in {shapefile_path}")
                continue  # Skip this shapefile if no ID column is found

            print(f"Using '{id_column}' as the ID column in {shapefile_path}")

            # Check the geometry type and handle conversion if needed
            geometry_type = gdf.geometry.geom_type.unique()
            print("Unique Geometry Types:", geometry_type)

            # If geometry is Point and we want MultiPolygon, convert it
            if 'Point' in geometry_type:
                print(f"Converting 'Point' geometry to 'MultiPolygon' in {shapefile_path}")
                gdf['geometry'] = gdf['geometry'].apply(lambda x: Polygon([x.coords]) if isinstance(x, Point) else x)
                gdf = gdf.set_geometry('geometry')

            # Iterate through rows and insert into the database
            for _, row in gdf.iterrows():
                loc_id = row[id_column]  # Use the dynamically found ID column

                # Check for None values in ID or geometry columns
                if loc_id is None or row['geometry'] is None:
                    print(f"Skipping row with missing values: {row}")
                    continue  # Skip rows with missing values

                # Convert the geometry to WKT format, so it can be inserted as a geometry
                geom = row['geometry'].wkt  # Convert the geometry to WKT

                # Prepare the data to insert, including the category
                shape = ShapeFile(loc_id=loc_id, wkb_geometry=geom, category=category)

                try:
                    # Insert the record into the database
                    session.add(shape)
                    session.commit()
                except SQLAlchemyError as e:
                    # If there's an error, print it and skip this record
                    print(f"Error inserting record {loc_id}: {e}")
                    session.rollback()  # Rollback the session to prevent other issues
                    continue  # Skip this record and move on

            print(f"Processed {shapefile_path}")

# Close the session
session.close()

print("All shapefiles processed successfully.")
