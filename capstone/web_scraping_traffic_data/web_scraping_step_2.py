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

# Reads in excel file of links to download
# each file contains all links for a single sensor

# Setup Selenium
options = Options()
options.add_experimental_option("detach", True)
driver = webdriver.Chrome(options=options)

# Base URLs
BASE_URL = "https://mhd.public.ms2soft.com/tcds/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Create output folder
os.makedirs("downloads", exist_ok=True)

def go_to_search_page():
    driver.get("https://mhd.public.ms2soft.com/tcds/tsearch.asp?loc=Mhd&mod=TCDS")
    iframe = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "iframe")))
    driver.switch_to.frame(iframe)

# Load initial search page
go_to_search_page()

for filename in os.listdir('links'):
    links_df = pd.read_excel('links/'+filename)
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
            print(f"🌐 [{idx}] Getting detail page: {detail_url}")
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
                file = f"{filename[:-5]}_{idx}.xlsx"
                filepath = os.path.join("downloads", file)
                with open(filepath, "wb") as f:
                    f.write(file_res.content)
                print(f"✅ Downloaded: {file}")
            else:
                print(f"❌ Failed to download Excel file: {excel_href} - Status: {file_res.status_code}")

        except Exception as e:
            print(f"❌ Exception on {detail_url}: {e}")