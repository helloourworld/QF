# -*- coding: utf-8 -*-
"""
@author: lyu

crontab -e

0 0 * * 0 /usr/bin/python3 /path/QF/P_QF/FunCode/utils_1_SyncBookmarks.py >> ~/Documents/sync_bookmarks.log 2>&1

crontab -l
"""
import sqlite3
import os
import datetime

# Path to Safari bookmarks file
safari_bookmarks_path = os.path.expanduser("~/Library/Safari/Bookmarks.db")

# Path to Chrome bookmarks file
chrome_bookmarks_path = os.path.expanduser("~/Library/Application Support/Google/Chrome/Default/Bookmarks")

# Path to log file
log_file_path = os.path.expanduser("~/Documents/sync_bookmarks.log")

# Connect to Safari bookmarks database
safari_conn = sqlite3.connect(safari_bookmarks_path)
safari_cursor = safari_conn.cursor()

# Get all bookmarks from Safari
safari_cursor.execute("SELECT title, url FROM bookmarks")
safari_bookmarks = safari_cursor.fetchall()

# Close Safari bookmarks database connection
safari_conn.close()

# Connect to Chrome bookmarks file
chrome_conn = sqlite3.connect(chrome_bookmarks_path)
chrome_cursor = chrome_conn.cursor()

# Delete all existing bookmarks in Chrome
chrome_cursor.execute("DELETE FROM bookmarks")

# Insert all Safari bookmarks into Chrome
for bookmark in safari_bookmarks:
    chrome_cursor.execute("INSERT INTO bookmarks (id, type, url, title, date_added, date_modified, meta_info, parent_id, position) VALUES (NULL, 1, ?, ?, strftime('%s', 'now'), strftime('%s', 'now'), NULL, 1, 0)", bookmark)

# Commit changes to Chrome bookmarks file
chrome_conn.commit()

# Close Chrome bookmarks file connection
chrome_conn.close()

# Write output to log file
with open(log_file_path, "a") as log_file:
    log_file.write(f"{datetime.datetime.now()}: Safari bookmarks have been synced with Google Chrome!\n")

print("Done!")
