import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Kaufman's Adaptive Moving Average (KAMA) Function
def calculate_kama(price, n=10, fast=2, slow=30):
    change = abs(price - price.shift(n))
    volatility = abs(price - price.shift(1)).rolling(n).sum()
    er = change / volatility
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc)**2
    
    kama = np.zeros(len(price))
    for i in range(len(price)):
        if i < n:
            kama[i] = price.iloc[i]
        else:
            kama[i] = kama[i-1] + sc.iloc[i] * (price.iloc[i] - kama[i-1])
    return pd.Series(kama, index=price.index)

# 2. Data Preparation
# Earnings Yield = Trailing 12-Month Earnings / Index Price.
data = pd.read_excel('./FunCode/US Stock Market Chart Watch/SP500.xlsx', index_col=0, parse_dates=True)  # Uncomment when you have the data

data['SP500_Index'] = data.iloc[:, 0]  # Assuming the first column is the S&P 500 Index
sp500 = data['SP500_Index'].dropna()

# Calculate AMA
ama = calculate_kama(sp500)

# 3. Signaling Logic & Regime Return Tracking
signals = pd.Series(0, index=sp500.index)
regime_start_indices = [] # To track when a regime flips
regime_type = []          # To track if it was Bull or Bear

current_regime = "Bearish"
peak = ama.iloc[0]
trough = ama.iloc[0]
last_flip_idx = 0

for i in range(1, len(ama)):
    current_val = ama.iloc[i]
    
    if current_regime == "Bearish":
        if current_val < trough: trough = current_val
        if current_val >= trough * 1.02: # 2% Recovery
            current_regime = "Bullish"
            peak = current_val
            last_flip_idx = i
            regime_start_indices.append(i)
            regime_type.append("Bullish")
            
    elif current_regime == "Bullish":
        if current_val > peak: peak = current_val
        if current_val <= peak * 0.95: # 5% Drop
            current_regime = "Bearish"
            trough = current_val
            last_flip_idx = i
            regime_start_indices.append(i)
            regime_type.append("Bearish")
            
    signals.iloc[i] = 1 if current_regime == "Bullish" else 0

# Calculate the return of the LATEST regime
latest_start_idx = regime_start_indices[-1]
latest_start_price = sp500.iloc[latest_start_idx]
current_price = sp500.iloc[-1]
latest_return = ((current_price / latest_start_price) - 1) * 100
latest_regime_name = regime_type[-1]

# 4. Visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

# Top Clip: S&P 500 Index
ax1.plot(sp500.index, sp500, color='black', lw=1.2, label='S&P 500 Index')
ax1.set_title("S&P 500 INDEX vs ADAPTIVE MOVING AVERAGE MODEL", fontweight='bold')
ax1.set_yscale('log')

# Highlight Regimes
ax1.fill_between(sp500.index, ax1.get_ylim()[0], ax1.get_ylim()[1], 
                 where=(signals == 1), color='green', alpha=0.15, label='Bullish Regime')

# --- CUMULATIVE RETURN OF RECENT REGIME OVERLAY ---
info_text = (f"CURRENT REGIME: {latest_regime_name}\n"
             f"Started: {sp500.index[latest_start_idx].strftime('%Y-%m-%d')}\n"
             f"Regime Return: {latest_return:+.2f}%")

# Place the text box in the upper left of the top chart
props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='navy')
ax1.text(0.02, 0.95, info_text, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=props, fontweight='bold', color='navy')

# Bottom Clip: AMA
ax2.plot(ama.index, ama, color='blue', lw=1.5, label='Adaptive Moving Average')
ax2.set_title("Adaptive Moving Average (AMA) - (2% Bottom / 5% Peak Filter)")

# Annotate historical returns (Mode Boxes)
# These are representative placeholders for the annualized data mentioned in your prompt
ax2.text(0.02, 0.1, "Annualized Returns:\nBullish Regimes: +14.2%\nBearish Regimes: -4.8%", 
         transform=ax2.transAxes, bbox=dict(facecolor='grey', alpha=0.1))

plt.tight_layout()
plt.show()