import yfinance as yf
import pandas as pd
import numpy as np

# 1. Define Portfolios
current_portfolio = {
    'MSFT': 0.1668,  # a classic battle between investing in a great company and managing a healthy portfolio

    'XDIV.TO': 0.1420, 'NVDA': 0.1201,
    'CLML.TO': 0.1058, 'XGD.TO': 0.0991, 'CNQ.TO': 0.0942,
    'XHC.TO': 0.0683, 'CGL.TO': 0.0616,
    'AVGO': 0.0609,  # higher-growth engines
    'AMZN': 0.0336,  # higher-growth engines
    'FBTC': 0.0294, 'RBNK.TO': 0.0181
}

# Proposed weights based on March 5 analysis
optimized_portfolio = {
    'MSFT': 0.0800, 'XDIV.TO': 0.1500, 'NVDA': 0.1200,
    'CLML.TO': 0.1500, 'XGD.TO': 0.0500, 'CNQ.TO': 0.1000,
    'XHC.TO': 0.0700, 'CGL.TO': 0.0700, 'AVGO': 0.0800,
    'AMZN': 0.0800, 'FBTC': 0.0500, 'RBNK.TO': 0.0000
}

tickers = list(current_portfolio.keys())

# 2. Download Data
print("Pulling 3-year performance data...")
data = yf.download(tickers, period="3y")['Close']
returns = data.pct_change().dropna()

# 3. Calculation Function


def calculate_metrics(weights_dict, returns):
    w = np.array(list(weights_dict.values()))
    print(w.sum())  # Should be 1.0 or 100%
    # Annualized Return
    p_ret = np.sum(returns.mean() * w) * 252
    # Annualized Volatility
    p_vol = np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w)))
    # Sharpe Ratio (Assuming 4.25% Risk Free Rate in 2026)
    sr = (p_ret - 0.0425) / p_vol
    return p_ret, p_vol, sr


# 4. Results
curr_ret, curr_vol, curr_sr = calculate_metrics(current_portfolio, returns)
opt_ret, opt_vol, opt_sr = calculate_metrics(optimized_portfolio, returns)

print("\n" + "="*45)
print(f"{'Metric':<20} | {'Current':<10} | {'Optimized'}")
print("-" * 45)
print(f"{'Annual Return':<20} | {curr_ret*100:6.2f}%    | {opt_ret*100:6.2f}%")
print(f"{'Annual Volatility':<20} | {curr_vol*100:6.2f}%    | {opt_vol*100:6.2f}%")
print(f"{'Sharpe Ratio':<20} | {curr_sr:6.2f}     | {opt_sr:6.2f}")
print("="*45)

"""
=============================================
Metric               | Current    | Optimized
---------------------------------------------
Annual Return        |  36.72%    |  39.18%
Annual Volatility    |  19.81%    |  19.78%
Sharpe Ratio         |   1.64     |   1.77
=============================================

Investment Policy Statement (IPS) for RRSP
Account: RRSP (Registered Retirement Savings Plan)
Date: XXX
Investment Objective: Long-term capital appreciation (Growth) with a focus on maximizing the Sharpe Ratio (Risk-adjusted return).
Component	Policy Detail
Target Return	12% - 15% Annualized
Risk Tolerance	Moderate-Aggressive (Maximum Drawdown Target: <20%)
Core Theme	The AI Infrastructure Super-Cycle. Exposure to both the "Brains" (NVDA/AVGO) and the "Power" (CLML/CNQ).
Currency Strategy	USD Asset Neutrality. hold US-listed assets (NVDA, AVGO, MSFT) directly in USD to avoid the 15% withholding tax and CAD-hedging costs.
Rebalancing Frequency	Quarterly or when a position drifts +/- 5% from target.
Asset Constraints	No single stock > 15%. No "Yield Trap" ETFs (e.g., NVDY).
"""
"""Opportunity Cost: You are currently underweight in Amazon (3.36%) and Broadcom (6.09%). 
In early 2026, these two are growing faster than Microsoft. 
Trimming MSFT allows you to "fuel" these higher-growth engines."""