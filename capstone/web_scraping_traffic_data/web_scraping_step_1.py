from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import time
import os
import pandas as pd

# Setup Selenium
options = Options()
options.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=options)

# Base URLs
BASE_URL = "https://mhd.public.ms2soft.com/tcds/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Create output folder
os.makedirs("downloads", exist_ok=True)

# Prepare summary data collector
scrape_summary = []

def go_to_search_page():
    driver.get("https://mhd.public.ms2soft.com/tcds/tsearch.asp?loc=Mhd&mod=TCDS")
    iframe = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))
    driver.switch_to.frame(iframe)

# Load initial search page
go_to_search_page()
original_window = driver.current_window_handle

# For testing, use a small sample or 1 location ID
loc_ids_df = pd.read_excel('class_7_ids.xlsx', header=1)
loc_ids = loc_ids_df['Id'].tolist()

for loc_id in loc_ids:
    # only use if value is in 90's
    # loc_id = '0'+str(loc_id)
    print(f"\n🔍 Processing Loc ID: {loc_id}")
    all_links = []
    seen_links = set()
    offset = 0
    no_new_count = 0

    # Always go back to search page and switch iframe
    driver.switch_to.window(original_window)
    go_to_search_page()
    time.sleep(1)

    # Input the Loc ID
    input_box = WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.ID, "ddlLocalId"))
    )
    input_box.clear()
    input_box.send_keys(loc_id)
    time.sleep(1)

    # Click search
    search_button = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.ID, "btnSubmit"))
    )
    driver.execute_script("arguments[0].click();", search_button)
    time.sleep(2)

    last_link_count = -1
    stagnant_count = 0
    while True:
        try:
            # Try waiting for volume links to appear
            WebDriverWait(driver, 10).until(
                EC.presence_of_all_elements_located((By.XPATH, "//a[contains(@href, \"count_type='VOLUME'\")]"))
            )
        except Exception as e:
            print(f"⚠️ Failed to locate volume links: {e}")
            print("🔁 Attempting to click the 'Older' buttons to continue...")

            # Attempt clicking the first 'Older' button
            try:
                button_one = driver.find_element(
                    By.XPATH,
                    "//input[@type='button' and @title='Older' and contains(@onclick, \"ShowTable('TCDS_TDETAIL_AADT_DIV','tcds_tdetail_aadt','?\")]"
                )
                driver.execute_script("arguments[0].click();", button_one)
                print("➡️ Clicked the first 'Older' button (TCDS_TDETAIL_AADT_DIV)")
                time.sleep(2)
                continue  # Skip to next iteration of the loop
            except Exception as e1:
                print(f"❌ Failed to click the first 'Older' button: {e1}")
            
            # Attempt clicking the second 'Older' button
            try:
                button_two = driver.find_element(
                    By.XPATH,
                    "//input[@type='button' and @title='Older' and contains(@onclick, \"ShowTable('TCDS_TDETAIL_VOL_DIV','tcds_tdetail_vol','?\")]"
                )
                driver.execute_script("arguments[0].click();", button_two)
                print("➡️ Clicked the second 'Older' button (TCDS_TDETAIL_VOL_DIV) first loop")
                time.sleep(2)
                continue  # Skip to next iteration of the loop
            except Exception as e2:
                print(f"❌ Failed to click the second 'Older' button: {e2}")
                print("🛑 Could not find any links or click older buttons. Exiting loop.")
                break  # Exit loop if no fallback worked

        # If we reach here, it means links were found
        page_links = driver.find_elements(By.XPATH, "//a[contains(@href, \"count_type='VOLUME'\")]")
        new_on_page = 0
        volume_hrefs = []
        for link in page_links:
            href = link.get_attribute("href")
            if not href.startswith("http"):
                href = f"{BASE_URL}{href}"
            if href not in seen_links:
                volume_hrefs.append(href)
                all_links.append(href)
                seen_links.add(href)
                new_on_page += 1
        print(f"len of list {len(all_links)}")
        save_list = pd.DataFrame(all_links,columns=['links'])
        filename = 'links/'+str(loc_id)+'_'+'all_links.xlsx'
        save_list.to_excel(filename)
        # Check if the link count has changed
        if len(all_links) == last_link_count:
            stagnant_count += 1
            print(f"🔁 Link count unchanged ({len(all_links)}). Stagnant count: {stagnant_count}")
        else:
            stagnant_count = 0
            last_link_count = len(all_links)

        # Exit loop if we've hit the same count twice in a row
        if stagnant_count >= 2:
            print("🛑 Link count unchanged for two consecutive iterations. Exiting loop.")
            break

        if new_on_page == 0:
            try:
                button_one = driver.find_element(
                    By.XPATH,
                    "//input[@type='button' and @title='Older' and contains(@onclick, \"ShowTable('TCDS_TDETAIL_AADT_DIV','tcds_tdetail_aadt','?\")]"
                )
                driver.execute_script("arguments[0].click();", button_one)
                print("➡️ Clicked the first 'Older' button (TCDS_TDETAIL_AADT_DIV) second loop")
                time.sleep(2)
                continue
            except Exception as e:
                print(f"❌ Failed to click the first button: {e}")
            
            try:
                button_two = driver.find_element(
                    By.XPATH,
                    "//input[@type='button' and @title='Older' and contains(@onclick, \"ShowTable('TCDS_TDETAIL_VOL_DIV','tcds_tdetail_vol','?\")]"
                )
                driver.execute_script("arguments[0].click();", button_two)
                print("➡️ Clicked the second 'Older' button (TCDS_TDETAIL_VOL_DIV)")
                time.sleep(2)
                continue
            except Exception as e:
                print(f"❌ Failed to click the second button: {e}")
                print("🛑 No more new links and failed to click any 'Older' buttons. Exiting loop.")
                break
