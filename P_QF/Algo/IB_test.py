#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
@file: IB_test.py.py 
Created on: 2025-03-11 3:42 p.m.
@author: yulijun
@gmail: davidyjun@gmail.com
@desc: 
"""
from ibapi.client import EClient
# Responsible for sending orders, receiving market data and managing
# the connection between IDE & IB API

from ibapi.wrapper import EWrapper
# Defines how your code should respond to events/data received from IB

import threading
# Threads are used to run API and TWS concurrently

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
# Combines EWrapper & EClient to interact with IB API

def run_loop():
    app.run()
# Starts the API and runs it in a separate thread
# Allows API and TWS to run concurrently

app = IBapi()
app.connect('127.0.0.1', 7497, 123)
# Establishes the connection to the API
# Substitute 192.168.1.120 for your own machine's IP
# Substitute 7496 for your own Socket port number if required.
# For paper accounts, the Socket port number is typically set to 7497
# Substitute 123 for your own client ID that you set earlier.

# Confirm successful connection
print("Connected to the API successfully.")

# Start the socket in a thread
api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()
# This starts the API