import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# This file creates generalized definitions for use_type, since there are over 2000
# different types. Might not be useful, as land use was not found to impact the model

engine = create_engine('postgresql+psycopg2://postgres:yourpassword@localhost/spatial_db')
Session = sessionmaker(bind=engine)
session = Session()

raw_use_types_df = pd.read_sql(f"""
    SELECT distinct use_type
    FROM general_data.shapefiles 
""", engine)

raw_use_types = raw_use_types_df['use_type'].tolist()


# Define broad categories with associated keywords
land_use_groups = {
    "Residential: Single Family": [
        "single family", "duplex"
    ],
    "Residential: Multi-Family": [
        "two-family", "three-family", "apartment", "condo", "housing",
        "affiliated housing", "apt", "four to eight", "residential condominium",
        "multiple houses"
    ],
    "Commercial: Retail": [
        "retail", "store", "shop", "supermarket", "restaurant", "eating",
        "drinking", "diner", "mall", "shopping", "discount", "improved selectman commercial"
    ],
    "Commercial: Office": [
        "office", "general office", "bank", "insurance"
    ],
    "Healthcare": [
        "hospital", "medic", "clinic", "rehab", "hosp", "medical",
        "charitable hospitals", "health"
    ],
    "School/Education": [
        "school", "college", "university", "educational", "day care", "child care"
    ],
    "Religious": [
        "church", "mosque", "synagogue", "temple", "rectory", "religious"
    ],
    "Industrial": [
        "industrial", "warehouse", "manufacturing", "distribution", "storage",
        "plant", "factory", "comm whse", "research and development facilities"
    ],
    "Transportation": [
        "mbta", "transit", "bus", "rail", "garage", "parking lot", "gas",
        "transportation", "car wash", "vehicle", "motor", "fuel service areas"
    ],
    "Recreational: Public": [
        "park", "conservation", "greenbelt", "recreational", "open space",
        "fishing", "hunting"
    ],
    "Recreational: Private": [
        "golf", "club", "gym", "athletic", "arena", "camping", "billiards",
        "archery", "swim", "sports"
    ],
    "Agricultural": [
        "agri", "farm", "orchard", "pasture", "crop", "cranberry", "field",
        "woodland", "ch61", "hort"
    ],
    "Government": [
        "municipal", "commonwealth", "massachusetts", "government", "county",
        "federal", "dmh", "doe", "dcr", "public", "selectmen", "city council",
        "housing authority", "hsng auth m94", "us govt", "dept of fish and game",
        "mass. highway dept.", "police", "town-prop", "dept. of fish and game, environmental law enforcement (dfg, formerly dfweele) (non-reimbursable)",
        "comm. of mass. (other, non-reimbursable)"
    ],
    "Vacant": [
        "vacant", "undevelopable", "undeveloped", "developable"
    ],
    "Utilities": [
        "electric", "gas", "sewage", "water", "pipeline", "telecommunication",
        "power", "generation"
    ],
    "Institutional/Charitable": [
        "charit", "non-profit", "fraternal", "youth", "community center",
        "function hall"
    ],
    "Hotels/Hospitality": [
        "hotel", "motel", "inn", "resort"
    ],
    "Auto Repair": [
        "auto repair facilities", "auto repair"
    ],
    "Mining/Quarry": [
        "sand and gravel mining", "quarry"
    ],
    "Cemeteries": [
        "cemetery", "cemeteries c"
    ],
    "Mixed Use": [
        "mixed use", "50/50", "split use", "primarily"
    ],
    "Other": [
        "unknown", "res aclnpo", "accessory land with improvement", "consominium m-05"
    ]  # fallback for uncategorized
}




def classify_use_type(use_type):
    use_lower = str(use_type).lower()
    for category, keywords in land_use_groups.items():
        if any(kw in use_lower for kw in keywords):
            return category
    return "Other"

# Apply classification
grouped_use_types = pd.DataFrame({
    "original": raw_use_types,
    "grouped": [classify_use_type(u) for u in raw_use_types]
})

# Print or save to file
print(grouped_use_types)
grouped_use_types.to_csv("grouped_land_use_types.csv", index=False)
