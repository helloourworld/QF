# -*- coding: utf-8 -*-
"""
@author: lyu

crontab -e

0 0 * * 0 /usr/bin/python3 /path/QF/P_QF/FunCode/utils_1_SyncBookmarks.py >> ~/Documents/sync_bookmarks.log 2>&1

crontab -l
"""

import scrapy
from scrapy.http import Request

class YieldSpider(scrapy.Spider):
    name = 'yields'
    allowed_domains = ['faa.gov']
    start_urls = ['https://www.faa.gov/data_research']

    def parse(self, response):
        self.logger.info("A response from %s just arrived!", response.url)
        # Your parsing logic here
        # For XLS or XLSX files, look for links with .xls or .xlsx extensions
        xls_links = response.css('a[href$=".xls"], a[href$=".xlsx"]::attr(href)').extract()
        for link in xls_links:
            yield Request(url=link, callback=self.save_xls_file)

    def save_xls_file(self, response):
        # Save the XLS file
        filename = response.url.split('/')[-1]  # Extract the filename from the URL
        with open(filename, 'wb') as f:
            f.write(response.body)
        self.log(f'Saved {filename}')

        
import pandas as pd

# Replace with the URL of the FAA data file (e.g., CY 2022 Passenger Boarding data)
data_url = "https://www.faa.gov/sites/faa.gov/files/2022-09/cy21-all-enplanements.xlsx"

# Read the Excel file into a pandas DataFrame
try:
    df = pd.read_excel(data_url)
    print("Data loaded successfully!")
    print(df.head())  # Display the first few rows of data
except Exception as e:
    print(f"Error loading data: {e}")
