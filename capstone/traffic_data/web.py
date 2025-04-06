from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd

# Setup options
options = Options()
options.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=options)

# Set up dataframe to store data
interval_df = pd.DataFrame(columns=[
    "Time", "1st", "2nd", "3rd", "4th", "Hourly count",
    "date", "Location id", "town", "weekday"
])

# STEP 1
# Navigate to main page
driver.get("https://mhd.public.ms2soft.com/tcds/tsearch.asp?loc=Mhd&mod=TCDS")

# Wait for and switch to iframe
iframe = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))
driver.switch_to.frame(iframe)

# Enter the location ID
input_box = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "ddlLocalId")))
input_box.clear()

#TODO: update to iterate through all ids
input_box.send_keys("0130003")
time.sleep(1)

# Click search
search_button = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "btnSubmit")))
driver.execute_script("arguments[0].click();", search_button)

#STEP 2: Open links in Volume Cont Table
time.sleep(2)
# Find all anchor tags with href containing "type=VOLUME"
volume_links = driver.find_elements(By.XPATH, "//a[contains(@href, \"count_type='VOLUME'\")]")

# Open each link in a new tab
for link in volume_links:
    href = link.get_attribute("href")
    if href:
        full_url = href if href.startswith("http") else f"https://mhd.public.ms2soft.com/tcds/{href}"
        driver.execute_script("window.open(arguments[0], '_blank');", full_url)
        time.sleep(1)
        try:
            #TODO: Fix this part of script
            # get data to save
            # Get metadata
            date_text = driver.find_element(By.ID, "lblCountDate").text.strip()
            location_id = driver.find_element(By.ID, "lblLocalId").text.strip()
            town = driver.find_element(By.ID, "lblLocation").text.strip()
            weekday = datetime.strptime(date_text, "%m/%d/%Y").strftime("%A")

            # Get the interval table rows
            table = driver.find_element(By.ID, "dgInterval")
            rows = table.find_elements(By.TAG_NAME, "tr")[1:]  # skip header row

            for row in rows:
                cells = row.find_elements(By.TAG_NAME, "td")
                if len(cells) >= 6:
                    interval_df.loc[len(interval_df)] = [
                        cells[0].text.strip(),  # Time
                        cells[1].text.strip(),  # 1st
                        cells[2].text.strip(),  # 2nd
                        cells[3].text.strip(),  # 3rd
                        cells[4].text.strip(),  # 4th
                        cells[5].text.strip(),  # Hourly count
                        date_text, location_id, town, weekday
                    ]
            breakpoint()
        except Exception as e:
            print(f"❌ Error processing tab: {e}")

# Output how many links were opened
print(f"✅ Opened {len(volume_links)} volume count pages.")



