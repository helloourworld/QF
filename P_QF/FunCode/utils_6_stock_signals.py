# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date

st.set_page_config(page_title="EMA Signals Explorer", layout="wide")

def fetch_data(symbol: str, start_date: date, adjust: bool):
    df = yf.download(symbol, start=start_date, auto_adjust=adjust)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(subset=["Close"])
    return df

def compute_emas_and_signals(df: pd.DataFrame, s:int, m:int, l:int):
    out = df.copy()
    out[f"ema_{s}"] = out["Close"].ewm(span=s, adjust=False).mean() # Short EMA
    out[f"ema_{m}"] = out["Close"].ewm(span=m, adjust=False).mean() # Mid EMA
    out[f"ema_{l}"] = out["Close"].ewm(span=l, adjust=False).mean() # Long EMA

    # Generate signals
    # EMA50 > EMA200 or Close > EMA200
    # and Close > EMA50 for bullish entry
    out["bullish"] = ((out[f"ema_{m}"] > out[f"ema_{l}"]) | (out["Close"] > out[f"ema_{l}"])) & (out["Close"] > out[f"ema_{m}"]) # Bullish condition
    out["entry_signal"] = out["bullish"] & (~out["bullish"].shift(1, fill_value=False))
    # Close < EMA20 for bearish exit
    out["bearish"] = out["Close"] < out[f"ema_{s}"] # Bearish condition
    out["exit_signal"] = out["bearish"] & (~out["bearish"].shift(1, fill_value=False))

    trades = []
    in_trade = False
    entry_date = None
    entry_price = None
    for idx, row in out.iterrows():
        if not in_trade and row["entry_signal"]:
            in_trade = True
            entry_date = idx
            entry_price = row["Close"]
        elif in_trade and row["exit_signal"]:
            exit_date = idx
            exit_price = row["Close"]
            trades.append({
                "Entry Date": entry_date,
                "Exit Date": exit_date,
                "Entry Price": float(round(entry_price, 2)),
                "Exit Price": float(round(exit_price, 2)),
                "Return": float(round(exit_price / entry_price - 1, 4))
            })
            in_trade = False
    if in_trade:
        exit_date = out.index[-1]
        exit_price = out.iloc[-1]["Close"]
        trades.append({
            "Entry Date": entry_date,
            "Exit Date": exit_date,
            "Entry Price": float(round(entry_price, 2)),
            "Exit Price": float(round(exit_price, 2)),
            "Return": float(round(exit_price / entry_price - 1, 4))
        })

    trades_df = pd.DataFrame(trades)
    return out, trades_df

def plot_with_signals(df: pd.DataFrame, s:int, m:int, l:int, ticker:str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df["Close"], mode="lines", name="Close", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=df.index, y=df[f"ema_{s}"], mode="lines", name=f"EMA {s}", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df.index, y=df[f"ema_{m}"], mode="lines", name=f"EMA {m}", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=df.index, y=df[f"ema_{l}"], mode="lines", name=f"EMA {l}", line=dict(color="red")))

    entries = df[df["entry_signal"]]
    exits = df[df["exit_signal"]]
    if not entries.empty:
        fig.add_trace(go.Scatter(x=entries.index, y=entries["Close"], mode="markers", marker=dict(color="green", symbol="triangle-up", size=10), name="Entry"))
    if not exits.empty:
        fig.add_trace(go.Scatter(x=exits.index, y=exits["Close"], mode="markers", marker=dict(color="orange", symbol="triangle-down", size=10), name="Exit"))

    fig.update_layout(title=f"{ticker} — EMA strategy", xaxis_title="Date", yaxis_title="Price")
    return fig

def run_streamlit_app():
    st.sidebar.header("Data & Strategy")
    ticker = st.sidebar.text_input("Ticker", value="GOOG")
    start_dt = st.sidebar.date_input("Start date", value=date(2024, 1, 1))
    auto_adjust = st.sidebar.checkbox("Auto-adjust prices", value=True)
    short_span = st.sidebar.number_input("Short EMA span", min_value=5, max_value=100, value=20)
    mid_span = st.sidebar.number_input("Mid EMA span", min_value=20, max_value=200, value=50)
    long_span = st.sidebar.number_input("Long EMA span", min_value=50, max_value=400, value=200)
    run = st.sidebar.button("Run analysis")

    st.title("EMA Crossover Signal Explorer")
    st.markdown("Configure parameters in the sidebar and press Run.")

    if run:
        with st.spinner("Downloading data and computing signals..."):
            data = fetch_data(ticker, start_dt, auto_adjust)
            if data.empty:
                st.error("No data returned for that ticker / date range.")
            else:
                processed, trades = compute_emas_and_signals(data, short_span, mid_span, long_span)
                fig = plot_with_signals(processed, short_span, mid_span, long_span, ticker)

                col1, col2 = st.columns([3,1])
                with col1:
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    st.metric("Data points", len(processed))
                    st.metric("Total trades", len(trades))
                    if not trades.empty:
                        total_return = (trades["Return"] + 1).prod() - 1
                        st.metric("Cumulative trade return", f"{total_return:.2%}")

                st.subheader("Recent Signals & Trades")
                st.dataframe(processed[[ "Close", f"ema_{short_span}", f"ema_{mid_span}", f"ema_{long_span}", "entry_signal", "exit_signal"]].tail(50))

                if not trades.empty:
                    st.subheader("Trade history")
                    st.dataframe(trades.sort_values("Entry Date", ascending=False).reset_index(drop=True))
                else:
                    st.info("No completed trades found for the selected parameters.")
    else:
        st.info("Adjust settings in the sidebar and click Run to start the analysis.")

def demo_matplotlib_ema(ticker="GOOG", start="2024-01-01"):
    df = yf.download(ticker, start=start)
    df.columns = df.columns.get_level_values(0)
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

    df['Bullish'] = ((df['EMA50'] > df['EMA200']) | (df['Close'] > df['EMA200'])) & (df['Close'] > df['EMA50'])
    df['BullishSignal'] = df['Bullish'] & (~df['Bullish'].shift(1).fillna(False))
    df['Bearish'] = (df['Close'] < df['EMA20'])
    df['ExitSignal'] = df['Bearish'] & (~df['Bearish'].shift(1).fillna(False))

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label=f'{ticker} Price', color='black')
    plt.plot(df.index, df['EMA20'], label='EMA 20', color='orange')
    plt.plot(df.index, df['EMA50'], label='EMA 50', color='blue')
    plt.plot(df.index, df['EMA200'], label='EMA 200', color='red')
    plt.scatter(df[df['BullishSignal']].index, df[df['BullishSignal']]['Close'], color='green', label='Bullish Entry Signal', marker='^', zorder=5)
    plt.scatter(df[df['ExitSignal']].index, df[df['ExitSignal']]['Close'], color='orange', label='Exit Signal', marker='v', s=100)
    plt.title(f'{ticker} Price with EMAs and Bullish Entry Signals')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def demo_plotly_ema(ticker="MRK", start="2022-01-01"):
    df = yf.download(ticker, start=start, auto_adjust=True)
    df.columns = df.columns.get_level_values(0)
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
    df['Signal'] = np.where(df['EMA_50'] > df['EMA_200'], 1, -1)
    df['Position'] = df['Signal'].diff()
    buy_signals = df[df['Position'] == 2]
    sell_signals = df[df['Position'] == -2]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_50'], mode='lines', name='EMA 50'))
    fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], mode='lines', name='EMA 200'))
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', marker=dict(color='green', size=10), name='Buy Signal'))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', marker=dict(color='red', size=10), name='Sell Signal'))
    fig.update_layout(title=f'EMA Crossover Strategy for {ticker}', xaxis_title='Date', yaxis_title='Price', legend_title='Legend')
    fig.show()

if __name__ == "__main__":
    # Uncomment the desired demo/test function to run it directly
    # demo_matplotlib_ema("GOOG", "2024-01-01")
    # demo_plotly_ema("MRK", "2022-01-01")
    run_streamlit_app()




