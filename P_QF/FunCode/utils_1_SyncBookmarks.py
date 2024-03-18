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

        
