import streamlit as st
import pandas as pd
import importlib.util

def ensure_installed(pkg_name):
    if importlib.util.find_spec(pkg_name) is None:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Could not install package {pkg_name}: {e}")

ensure_installed("yfinance")

import yfinance as yf
import requests
import talib
from datetime import date
import subprocess
import sys

def get_sp500_components():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    header = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest"
    }

    r = requests.get(url, headers=header)

    df = pd.read_html(r.text
                      )
    df = df[0]
    tickers = df["Symbol"].to_list()
    tickers_companies_dict = dict(zip(df["Symbol"], df["Security"]))
    return tickers, tickers_companies_dict


indicators = [
    "Simple Moving Average",
    "Exponential Moving Average",
    "Relative Strength Index",
]


def apply_indicator(indicator, data):
    if data is None or data.empty or "Close" not in data.columns:
        return pd.DataFrame()

    if indicator == "Simple Moving Average":
        sma = talib.SMA(data["Close"].values)
        return pd.DataFrame({"Close": data["Close"].values, "SMA": sma}, index=data.index)
    elif indicator == "Exponential Moving Average":
        ema = talib.EMA(data["Close"].values)
        return pd.DataFrame({"Close": data["Close"].values, "EMA": ema}, index=data.index)
    elif indicator == "Relative Strength Index":
        rsi = talib.RSI(data["Close"].values)
        return pd.DataFrame({"Close": data["Close"].values, "RSI": rsi}, index=data.index)

    return pd.DataFrame()


st.title("Stock Data Analysis")
st.write("A simple app to download stock data and apply technical analysis indicators.")

st.sidebar.header("Stock Parameters")

available_tickers, tickers_companies_dict = get_sp500_components()

ticker = st.sidebar.selectbox(
    "Ticker", available_tickers, format_func=tickers_companies_dict.get)

start = st.sidebar.date_input("Start date:", pd.Timestamp("2020-01-01"))
end = st.sidebar.date_input("End date:", date.today())
data = yf.download(ticker, start, end)

# print(data.head())
# If yfinance returned a MultiIndex (e.g., when downloading multiple tickers), flatten to the OHLCV level
if isinstance(data.columns, pd.MultiIndex):
    try:
        # level 1 typically contains 'Open', 'High', 'Low', 'Close', ...
        data.columns = data.columns.get_level_values(0)
    except Exception:
        # fallback to dropping the top level
        data.columns = data.columns.droplevel(1)

selected_indicator = st.selectbox(
    "Select a technical analysis indicator:", indicators)

indicator_data = apply_indicator(selected_indicator, data)

st.write(f"{selected_indicator} for {ticker}")
st.line_chart(indicator_data)

st.write("Stock data for", ticker)
st.dataframe(data.tail(10))
