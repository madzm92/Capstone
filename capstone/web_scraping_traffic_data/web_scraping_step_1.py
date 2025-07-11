from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import requests
from bs4 import BeautifulSoup
import time
import os
import pandas as pd

# # Load Loc IDs
# df = pd.read_excel("filtered_traffic_locations_list.xlsx")
# df = df[df['town_name'] == 'Acton ']
# loc_ids = df['location_id'].astype(str).tolist()

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
loc_ids = ['M4003S']
# loc_ids = ['RPA10-218-00009']

# loc_ids = [
#     "RPA10-218-00009", "RPA10-218-00010", "RPA10-247-00010", "RPA10-250-00014", "RPA10-250-00016", "RPA10-250-00017",
#     "RPA10-293-00019", "RPA10-293-00022", "RPA10-293-00028", "RPA10-293-00029", "RPA10-293-00030", "RPA10-293-00031",
#     "RPA10-310-00001", "RPA10-310-00022", "RPA10-310-00024", "RPA10-310-00026", "RPA10-310-00027", "RPA10-310-00029",
# ]
for loc_id in loc_ids:
    print(f"\n🔍 Processing Loc ID: {loc_id}")
    all_links = []
    seen_links = set()
    offset = 0
    no_new_count = 0

    try:
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

        # while True:
        #     try:
        #         # Try waiting for volume links to appear
        #         WebDriverWait(driver, 10).until(
        #             EC.presence_of_all_elements_located((By.XPATH, "//a[contains(@href, \"count_type='VOLUME'\")]"))
        #         )
        #     except Exception as e:
        #         print(f"⚠️ Failed to locate volume links: {e}")
        #         print("🔁 Attempting to click the 'Older' buttons to continue...")

        #         # Attempt clicking the first 'Older' button
        #         try:
        #             button_one = driver.find_element(
        #                 By.XPATH,
        #                 "//input[@type='button' and @title='Older' and contains(@onclick, \"ShowTable('TCDS_TDETAIL_AADT_DIV','tcds_tdetail_aadt','?\")]"
        #             )
        #             driver.execute_script("arguments[0].click();", button_one)
        #             print("➡️ Clicked the first 'Older' button (TCDS_TDETAIL_AADT_DIV)")
        #             time.sleep(2)
        #             continue  # Skip to next iteration of the loop
        #         except Exception as e1:
        #             print(f"❌ Failed to click the first 'Older' button: {e1}")
                
        #         # Attempt clicking the second 'Older' button
        #         try:
        #             button_two = driver.find_element(
        #                 By.XPATH,
        #                 "//input[@type='button' and @title='Older' and contains(@onclick, \"ShowTable('TCDS_TDETAIL_VOL_DIV','tcds_tdetail_vol','?\")]"
        #             )
        #             driver.execute_script("arguments[0].click();", button_two)
        #             print("➡️ Clicked the second 'Older' button (TCDS_TDETAIL_VOL_DIV) first loop")
        #             time.sleep(2)
        #             continue  # Skip to next iteration of the loop
        #         except Exception as e2:
        #             print(f"❌ Failed to click the second 'Older' button: {e2}")
        #             print("🛑 Could not find any links or click older buttons. Exiting loop.")
        #             break  # Exit loop if no fallback worked

        #     # If we reach here, it means links were found
        #     page_links = driver.find_elements(By.XPATH, "//a[contains(@href, \"count_type='VOLUME'\")]")
        #     new_on_page = 0
        #     volume_hrefs = []
        #     for link in page_links:
        #         href = link.get_attribute("href")
        #         if not href.startswith("http"):
        #             href = f"{BASE_URL}{href}"
        #         if href not in seen_links:
        #             volume_hrefs.append(href)
        #             all_links.append(href)
        #             seen_links.add(href)
        #             new_on_page += 1
        #     print(f"len of list {len(all_links)}")
        #     save_list = pd.DataFrame(all_links,columns=['links'])
        #     save_list.to_excel('all_links.xlsx')
        #     if new_on_page == 0:
        #         try:
        #             button_one = driver.find_element(
        #                 By.XPATH,
        #                 "//input[@type='button' and @title='Older' and contains(@onclick, \"ShowTable('TCDS_TDETAIL_AADT_DIV','tcds_tdetail_aadt','?\")]"
        #             )
        #             driver.execute_script("arguments[0].click();", button_one)
        #             print("➡️ Clicked the first 'Older' button (TCDS_TDETAIL_AADT_DIV) second loop")
        #             time.sleep(2)
        #             continue
        #         except Exception as e:
        #             print(f"❌ Failed to click the first button: {e}")
                
        #         try:
        #             button_two = driver.find_element(
        #                 By.XPATH,
        #                 "//input[@type='button' and @title='Older' and contains(@onclick, \"ShowTable('TCDS_TDETAIL_VOL_DIV','tcds_tdetail_vol','?\")]"
        #             )
        #             driver.execute_script("arguments[0].click();", button_two)
        #             print("➡️ Clicked the second 'Older' button (TCDS_TDETAIL_VOL_DIV)")
        #             time.sleep(2)
        #             continue
        #         except Exception as e:
        #             print(f"❌ Failed to click the second button: {e}")
        #             print("🛑 No more new links and failed to click any 'Older' buttons. Exiting loop.")
        #             break

        links_df = pd.read_excel('all_links.xlsx')
        breakpoint()
        all_links = links_df['links'].tolist()
        
        # Use Selenium cookies in requests session
        selenium_cookies = driver.get_cookies()
        session = requests.Session()
        for cookie in selenium_cookies:
            session.cookies.set(cookie['name'], cookie['value'])

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }

        for idx, detail_url in enumerate(all_links):
            try:
                print(f"🌐 [{loc_id}] Getting detail page: {detail_url}")
                res = session.get(detail_url, headers=headers, allow_redirects=True)
                if res.status_code != 200:
                    print(f"❌ Failed to load detail page: {detail_url} - Status: {res.status_code}")
                    continue

                soup = BeautifulSoup(res.text, "html.parser")
                excel_a_tag = soup.select_one("ul.btnTCnt li a[href*='rpt_volume_count.aspx']")

                if not excel_a_tag:
                    print(f"❌ Excel download link not found for: {detail_url}")
                    continue

                # Construct full Excel file URL
                excel_href = excel_a_tag["href"]
                if not excel_href.startswith("http"):
                    excel_href = f"{BASE_URL}{excel_href}"

                # Use same session to download file
                file_res = session.get(excel_href, headers=headers, allow_redirects=True)
                if file_res.status_code == 200:
                    filename = f"acton_{loc_id}_{idx}.xls"
                    filepath = os.path.join("downloads", filename)
                    with open(filepath, "wb") as f:
                        f.write(file_res.content)
                    print(f"✅ Downloaded: {filename}")
                else:
                    print(f"❌ Failed to download Excel file: {excel_href} - Status: {file_res.status_code}")

            except Exception as e:
                print(f"❌ Exception on {detail_url}: {e}")

        # Record summary
        scrape_summary.append({
            'location_id': loc_id,
            # 'town_name': town_name,  # Uncomment if town_name is defined earlier
            'total_volume_records': len(all_links),
            'total_pages_checked': offset
        })

    except:
        continue