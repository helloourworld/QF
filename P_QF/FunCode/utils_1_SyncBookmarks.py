# -*- coding: utf-8 -*-
"""
@author: lyu

crontab -e

0 0 * * 0 /usr/bin/python3 /path/QF/P_QF/FunCode/utils_1_SyncBookmarks.py >> ~/Documents/sync_bookmarks.log 2>&1

crontab -l
"""
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
