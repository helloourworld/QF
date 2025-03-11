#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
@file: IB_AccInfo.py.py 
Created on: 2025-03-11 3:29 p.m.
@author: yulijun
@gmail: davidyjun@gmail.com
@desc: 
"""
from ibapi.client import *
from ibapi.wrapper import *
from ibapi.contract import Contract
import time


class TradeApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def accountSummary(self, reqId: int, account: str, tag: str, value: str, currency: str):
        print("AccountSummary. ReqId:", reqId, "Account:", account, "Tag: ", tag, "Value:", value, "Currency:",
              currency)

    def accountSummaryEnd(self, reqId: int):
        print("AccountSummaryEnd. ReqId:", reqId)


app = TradeApp()
app.connect("127.0.0.1", 7497, clientId=1)

time.sleep(1)

app.reqAccountSummary(9001, "All", 'NetLiquidation')
app.run()