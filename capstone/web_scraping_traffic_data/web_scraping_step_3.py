from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

# Directory containing the files
data_dir = Path("downloads")

# Function to extract data from a single file
def extract_data_from_file(file_path):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # Extract metadata
    start_date_text = soup.find("td", string="Start Date")
    start_date = start_date_text.find_next_sibling("td").text.strip() if start_date_text else None

    town_text = soup.find("td", string="Community")
    town = town_text.find_next_sibling("td").text.strip() if town_text else None

    loc_id_text = soup.find("td", string="Location ID")
    loc_id = loc_id_text.find_next_sibling("td").text.strip() if loc_id_text else None

    # Convert start_date to datetime and extract weekday
    if start_date:
        try:
            start_date_dt = datetime.strptime(start_date, "%m/%d/%Y")
            weekday = start_date_dt.strftime("%A")
        except ValueError:
            start_date_dt = None
            weekday = None
    else:
        start_date_dt = None
        weekday = None

    # Extract interval data
    table = soup.find("th", string="Interval: 15 mins")
    
    if not table:
        table = soup.find("th", string="Interval: 60 mins")
        if not table:
            return pd.DataFrame()

    interval_table = table.find_parent("table")
    rows = interval_table.find_all("tr")[4:]  # Skip headers

    data = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) == 6 and "TOTAL" not in cols[0].text:
            time = cols[0].text.strip()
            values = [col.text.strip() for col in cols[1:]]
            data.append([time] + values + [start_date_dt, town, weekday, loc_id])
        if len(cols) == 2 and "TOTAL" not in cols[0].text:
            time = cols[0].text.strip()
            values = [col.text.strip() for col in cols[1:]]
            data.append([time] + [0,0,0,0] + [values[0]] + [start_date_dt, town, weekday, loc_id])

    columns = ["Time", "1st", "2nd", "3rd", "4th", "Hourly count", "Date", "Town", "Weekday", "Loc ID"]
    return pd.DataFrame(data, columns=columns)

# Iterate over all files in the folder
all_dfs = []
for file in data_dir.glob("*.xlsx"):
    df = extract_data_from_file(file)
    if not df.empty:
        all_dfs.append(df)

# Combine all data
combined_df = pd.concat(all_dfs, ignore_index=True)
combined_df.head()
breakpoint()
#TODO: drop index column
combined_df.to_excel('traffic_data_class_7_5.xlsx')