from sqlalchemy import create_engine, text
from capstone.database_set_up.table_definitions.traffic_tables import Base

# create table & insert data
DB_URI = "postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db"

# Set up SQLAlchemy and GeoAlchemy2
engine = create_engine(DB_URI)

with engine.connect() as conn:
    conn.execute(text("CREATE SCHEMA IF NOT EXISTS general_data;"))  # Create schema
    conn.commit()

Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)