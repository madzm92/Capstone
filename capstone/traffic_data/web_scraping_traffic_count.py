from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import pandas as pd

# STEP 0: Load the first 10 Loc IDs from the Excel file
df = pd.read_excel("filtered_traffic_locations_list.xlsx")
loc_ids = df['Loc ID'].astype(str).iloc[2000:5000].tolist()

# Setup options
options = Options()
options.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=options)

# Navigate to main page
def go_to_search_page():
    driver.get("https://mhd.public.ms2soft.com/tcds/tsearch.asp?loc=Mhd&mod=TCDS")
    iframe = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))
    driver.switch_to.frame(iframe)

# Initial load of the search page
go_to_search_page()
original_window = driver.current_window_handle

# Loop through each Loc ID
for loc_id in loc_ids:
    print(f"🔍 Searching for Loc ID: {loc_id}")

    try:
        # Input the Loc ID
        input_box = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, "ddlLocalId")))
        input_box.clear()
        input_box.send_keys(loc_id)
        time.sleep(1)

        # Click search
        search_button = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "btnSubmit")))
        driver.execute_script("arguments[0].click();", search_button)

        # Wait for results to load
        time.sleep(2)

        # Get all volume count links
        volume_links = driver.find_elements(By.XPATH, "//a[contains(@href, \"count_type='VOLUME'\")]")
        volume_hrefs = [
            link.get_attribute("href") if link.get_attribute("href").startswith("http")
            else f"https://mhd.public.ms2soft.com/tcds/{link.get_attribute('href')}"
            for link in volume_links
        ]

        # Loop through each stored URL
        for idx, full_url in enumerate(volume_hrefs):
            driver.execute_script("window.open(arguments[0]);", full_url)
            time.sleep(2)

            try:
                driver.switch_to.window(driver.window_handles[-1])

                # Wait for the Excel download link
                excel_link = WebDriverWait(driver, 15).until(
                    EC.element_to_be_clickable((
                        By.XPATH,
                        "//ul[contains(@class, 'btnTCnt')]/li/a[contains(@href, 'rpt_volume_count.aspx')]"
                    ))
                )

                excel_href = excel_link.get_attribute("href")
                if not excel_href.startswith("http"):
                    excel_href = f"https://mhd.public.ms2soft.com/tcds/{excel_href}"

                # Open Excel download link in a new tab
                driver.execute_script("window.open(arguments[0]);", excel_href)
                print(f"📥 [{loc_id}] Opened Excel link: {excel_href}")

                time.sleep(2)

            except Exception as e:
                print(f"❌ Error on volume tab for Loc ID {loc_id}: {e}")

            finally:
                driver.close()
                driver.switch_to.window(original_window)
                time.sleep(1)

        print(f"✅ Completed Loc ID: {loc_id} with {len(volume_hrefs)} volume links.")

    except Exception as e:
        print(f"❌ Error processing Loc ID {loc_id}: {e}")

    # Reset the page (ensure we're always on the main page for the next Loc ID)
    try:
        driver.switch_to.window(original_window)
        go_to_search_page()
    except Exception as e:
        print(f"⚠️ Failed to reset page for next Loc ID: {e}")

    time.sleep(2)

print("🏁 All Loc IDs processed.")
