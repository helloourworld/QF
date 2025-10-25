import scipy as sp
import numpy as np
import pandas as pd
import empyrical as ep
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

# Generate sample data
symbols = [
    "FCLS.NE",
    # "AMZN",
    # "GOOG",
    # "META",
    # "MSFT",
    # "NVDA",
    # "TSLA",
]
# [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
data = yf.download(symbols, start="2020-01-01",
                   end="2024-07-31", interval="1d")["Adj Close"]
# keep last month end data
# Resample to get the last business day of each month
last_month_end_data = data.resample('M').last()[:-1]

returns = data.pct_change().dropna()

if len(symbols) == 1:
    portfolio_returns = returns
else:
    portfolio_returns = returns.sum(axis=1)

print("Annual return: ", ep.annual_return(portfolio_returns, period="daily"))
print("Annual Volatility: ", ep.annual_volatility(
    portfolio_returns, period="daily"))

print("max dd: ", ep.max_drawdown(portfolio_returns))
print("CVaR: ", ep.conditional_value_at_risk(portfolio_returns, cutoff=0.10))
# Compute the Calmar Ratio using Empyrical
calmar_ratio = ep.calmar_ratio(portfolio_returns, period="daily")
print(f"Calmar Ratio: {calmar_ratio}")

# Compute the Sortino Ratio using Empyrical
calmar_ratio = ep.sortino_ratio(portfolio_returns, period="daily")
print(f"Sortino Ratio: {calmar_ratio}")

print("Calendar year and trailing returns",
      ep.aggregate_returns(portfolio_returns, convert_to="yearly"))
"""
The negative Calmar Ratio tells us this portfolio has a negative average annual return
 when compared to its maximum drawdown. In other words, for every unit of risk, 
 the return is not only insufficient but actually negative.
"""
print("Sharpe Ratio: ", ep.sharpe_ratio(portfolio_returns, period="daily"))
S = sp.stats.skew(portfolio_returns)
K = sp.stats.kurtosis(portfolio_returns)
SR = ep.sharpe_ratio(portfolio_returns, period="daily")
ASR = SR * (1 + S/6 * SR - (K-3)/24 * SR**2)
print("Adjusted Sharpe Ratio: ", ASR)

"""
Adjusted Sharpe Ratio adjusts Sharpe Ratio by incorporating a penalty factor for negative
skewness and positive excess kurtosis. If Skewness is negative and Excess Kurtosis is
positive the Adjusted Sharpe Ratio gets smaller compared to Sharpe Ratio.
If the returns are normally distributed, then the formula for the Adjusted 
Risk Return yields the same value as the traditional Sharpe Ratio.
"""
