from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.common import TickAttrib
from ibapi.contract import Contract
from ibapi.order import Order
import time
import threading

class IBWrapper(EWrapper):
    def connectionClosed(self):
        print("Connection closed")
    
    def nextValidOrderId(self, orderId: int):
        print(f"Next valid order ID: {orderId}")

class IBClient(EClient):
    def __init__(self, wrapper):
        EClient.__init__(self, wrapper)

def test_ib_connection():
    """Test connection to IB paper trading account"""
    app = IBClient(IBWrapper())
    
    # Connect to IB Gateway/TWS
    # clientId should be unique, useSSL=True for live
    app.connect("127.0.0.1", 4002, clientId=1)
    
    print("Attempting to connect to IB paper account...")
    
    # Start the connection in a thread
    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    
    time.sleep(2)
    
    if app.isConnected():
        print("✓ Successfully connected to IB paper account")
        app.disconnect()
    else:
        print("✗ Failed to connect - ensure TWS/Gateway is running on port 7497")

if __name__ == "__main__":
    test_ib_connection()