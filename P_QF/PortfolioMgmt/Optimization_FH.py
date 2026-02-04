import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# 1. DEFINE YOUR UNIVERSE
tickers = {
    'MSFT': 'Microsoft (CDR/Hedged)',
    'NVDA': 'NVIDIA (CDR/Hedged)',
    'AVGO': 'Broadcom (AI Diversification)',
    'CNQ.TO': 'CNRL (Energy/CAD Hedge)',
    'CLML.TO': 'Climate Leaders (Growth)',
    'FBTC.TO': 'Fidelity Bitcoin',
    'XDIV.TO': 'Canadian Dividend',
    'QSR.TO': 'Restaurant Brands (Defensive)',
    'CGL.TO': 'Gold Bullion (Insurance)',
    'OTEX.TO': 'OpenText (Value)'
}

symbols = list(tickers.keys())
risk_free_rate = 0.0428  # Current 10-Year US Treasury Yield (Feb 4, 2026)

# 2. DOWNLOAD DATA
print(f"Fetching 3 years of historical data for {len(symbols)} assets...")
data = yf.download(symbols, period="3y")['Close']
returns = data.pct_change().dropna()

# 3. OPTIMIZATION FUNCTIONS (Including Risk-Free Rate)
def get_portfolio_stats(weights, returns, rf_rate):
    weights = np.array(weights)
    # Annualized Return
    port_ret = np.sum(returns.mean() * weights) * 252
    # Annualized Volatility
    port_vol = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    # Sharpe Ratio (The core metric)
    sharpe = (port_ret - rf_rate) / port_vol
    return port_ret, port_vol, sharpe

def neg_sharpe(weights, returns, rf_rate):
    # We minimize the negative Sharpe to find the maximum
    return get_portfolio_stats(weights, returns, rf_rate)[2] * -1

# 4. CONSTRAINTS & BOUNDS
# Total weights must sum to 100%
cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
# Min 2% to ensure diversification, Max 15% to prevent over-concentration
bounds = tuple((0.02, 0.15) for _ in range(len(symbols)))
init_guess = [1/len(symbols)] * len(symbols)

# 5. RUN OPTIMIZATION
print("Optimizing for Maximum Sharpe Ratio...")
opt_results = minimize(
    neg_sharpe, 
    init_guess, 
    args=(returns, risk_free_rate), 
    method='SLSQP', 
    bounds=bounds, 
    constraints=cons
)

# 6. RESULTS & FINAL ANALYSIS
optimized_weights = opt_results.x
final_ret, final_vol, final_sr = get_portfolio_stats(optimized_weights, returns, risk_free_rate)

print("\n" + "="*60)
print(f"FINAL OPTIMIZED PORTFOLIO (Risk-Free Rate: {risk_free_rate*100:.2f}%)")
print("="*60)

results_df = pd.DataFrame({
    'Ticker': symbols,
    'Name': [tickers[s] for s in symbols],
    'Weight (%)': (optimized_weights * 100).round(2)
}).sort_values(by='Weight (%)', ascending=False)

print(results_df.to_string(index=False))

print("-" * 60)
print(f"Annualized Expected Return: {final_ret*100:.2f}%")
print(f"Annualized Volatility:      {final_vol*100:.2f}%")
print(f"FINAL SHARPE RATIO:         {final_sr:.2f}")
print("="*60)

"""
============================================================
FINAL OPTIMIZED PORTFOLIO (Risk-Free Rate: 4.28%)
============================================================
 Ticker                          Name  Weight (%)
   NVDA           NVIDIA (CDR/Hedged)       15.00
   AVGO Broadcom (AI Diversification)       15.00
OTEX.TO              OpenText (Value)       15.00
XDIV.TO             Canadian Dividend       15.00
CLML.TO      Climate Leaders (Growth)       12.25
   MSFT        Microsoft (CDR/Hedged)        9.79
 CNQ.TO       CNRL (Energy/CAD Hedge)        7.86
 CGL.TO      Gold Bullion (Insurance)        6.10
FBTC.TO              Fidelity Bitcoin        2.00
 QSR.TO Restaurant Brands (Defensive)        2.00
------------------------------------------------------------
Annualized Expected Return: 38.97%
Annualized Volatility:      17.25%
FINAL SHARPE RATIO:         2.01
============================================================

Ticker	Optimizer Weight	My Suggested Weight	Reason
NVDA.NE	15.00%	12.50%	Slightly de-risk the hardware peak.
AVGO.NE	15.00%	12.50%	Keep it equal to NVDA.
MSFT.NE	6.25%	10.00%	MSFT is too high-quality to hold only 6%.
CLML.TO	10.46%	10.00%	Round number for AI Infrastructure.
OTEX.TO	15.00%	10.00%	Don't over-rely on a "Value" turnaround.
XDIV.TO	15.00%	15.00%	Keep. Essential for CAD income.
CGL.TO	10.83%	10.00%	Keep. The "Fortress" gold position.
CNQ.TO	8.45%	10.00%	Increase slightly for better Oil exposure.
FBTC.TO	2.00%	5.00%	2% is too small to matter. 5% is a "real" bet.
QSR.TO	2.00%	5.00%	Good defensive hedge for a recession.


"""

suggested_weights = {
    'NVDA': 0.1500,
    'AVGO': 0.1500,
    'OTEX.TO': 0.1500,
    'XDIV.TO': 0.1500,
    'CGL.TO': 0.1083,
    'CLML.TO': 0.1046,
    'CNQ.TO': 0.0845,
    'MSFT': 0.0625,
    'FBTC.TO': 0.0200,
    'QSR.TO': 0.0200
}
weights = np.array(list(suggested_weights.values()))

perf_ret, perf_vol, perf_sr = get_portfolio_stats(weights, returns, risk_free_rate)

# 5. OUTPUT RESULTS
print("\n" + "="*40)
print("PORTFOLIO PERFORMANCE TEST RESULTS")
print("="*40)
print(f"Total Weights:        {np.sum(weights)*100:.2f}%")
print(f"Annualized Return:    {perf_ret*100:.2f}%")
print(f"Annualized Volatility: {perf_vol*100:.2f}%")
print(f"Risk-Free Rate used:  {risk_free_rate*100:.2f}%")
print("-" * 40)
print(f"SHARPE RATIO:         {perf_sr:.2f}")
print("="*40)

# Interpretation
if perf_sr > 2:
    print("INTERPRETATION: Exceptional risk-adjusted return.")
elif perf_sr > 1:
    print("INTERPRETATION: Good performance relative to risk.")
else:
    print("INTERPRETATION: Sub-optimal; risk may outweigh the returns.")