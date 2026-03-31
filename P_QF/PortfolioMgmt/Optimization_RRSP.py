import yfinance as yf
import pandas as pd
import numpy as np

# Portfolio replacing COST with L.TO
monopoly_sharpe_portfolio = {
    'MSFT': 0.09, 'NVDA': 0.10, 'AVGO': 0.10, 'CLML.TO': 0.10,
    'CNQ.TO': 0.10, 'XDIV.TO': 0.10, 'CGL.TO': 0.10, 'XHC.TO': 0.10,
    'FLEM.TO': 0.08, 'GOOG': 0.08, 'ATD.TO': 0.05
}

tickers = list(monopoly_sharpe_portfolio.keys())
data = yf.download(tickers, period="3y")['Close']
returns = data.pct_change(fill_method='ffill').dropna()
rf = 0.0425 # 2026 Risk-Free Rate

def test_sharpe(weights_dict, returns, rf):
    w = np.array(list(weights_dict.values()))
    p_ret = np.sum(returns.mean() * w) * 252
    p_vol = np.sqrt(np.dot(w.T, np.dot(returns.cov() * 252, w)))
    return p_ret, p_vol, (p_ret - rf) / p_vol

ret, vol, sr = test_sharpe(monopoly_sharpe_portfolio, returns, rf)

print("\n" + "="*40)
print("SPOUSE RRSP: MONOPOLY-SAFETY RESULTS")
print("="*40)
print(f"Annualized Return:    {ret:.2%}")
print(f"Annualized Volatility: {vol:.2%}")
print(f"SHARPE RATIO:         {sr:.2f}")
print("="*40)

"""
Mandate: Capital Deployment. The primary goal is to remove the "Cash Drag" and "Yield Erosion" (NVDY) to capture the 2026 AI expansion.
Rebalancing Trigger: Any single position exceeding 15% (Targeting MSFT/NVDA).
Asset Quality: Only hold "Category A" names. Companies in the "Net Debt < 1x EBITDA" range (CNQ, MSFT, GOOG).
Currency Policy: Since this is an RRSP, use the 45% cash to buy US-listed AMZN and AVGO directly in USD. This avoids the .NE (CDR) management fees and provides "True" US exposure.

Ticker	Weight	Role
MSFT.NE	9.0%	Tech Anchor	High quality, slightly trimmed for EM
NVDA.NE	10.0%	AI Hardware	Core growth
AVGO.NE	10.0%	AI Networking	Essential infrastructure
CLML.TO	10.0%	AI Power	Top 2026 Theme
CNQ.TO	10.0%	Energy/Income	Oil/CAD Hedge
XDIV.TO	10.0%	Cdn Dividends	Income floor
CGL.TO	10.0%	Physical Gold	Lowered from 12% to make room
XHC.TO	10.0%	Healthcare	Defensive stabilizer
FLEM.TO	8.0%	Emerging Markets	The Long-Term "Profit" Play
GOOG.NE	8.0%	Search Value	Cheapest Mag 7
ATD.TO	5.0%	Global Retail	The "Fair Value" Compounder

"""