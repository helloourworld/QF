import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Generate/Load Data
data = pd.read_excel('./FunCode/US Stock Market Chart Watch/SP500.xlsx', index_col=0, parse_dates=True)  # Uncomment when you have the data

data['SP500_Index'] = data.iloc[:, 0]  # Assuming the first column is the S&P 500 Index
sp500 = data['SP500_Index'].dropna()

# 2. Calculate Indicators
ma21 = sp500.rolling(window=21).mean()
ma63 = sp500.rolling(window=63).mean()
ratio = ma21 / ma63

# 3. Signaling Logic
signals = pd.Series(0, index=sp500.index)
regime_start_indices = []
regime_types = []
current_regime = "Bearish"
peak, trough = ratio.dropna().iloc[0], ratio.dropna().iloc[0]

for i in range(64, len(ratio)):
    curr_ratio = ratio.iloc[i]
    if current_regime == "Bearish":
        if curr_ratio < trough: trough = curr_ratio
        if curr_ratio > trough: 
            current_regime, peak = "Bullish", curr_ratio
            regime_start_indices.append(i); regime_types.append("Bullish")
    elif current_regime == "Bullish":
        if curr_ratio > peak: peak = curr_ratio
        if curr_ratio <= peak * 0.97:
            current_regime, trough = "Bearish", curr_ratio
            regime_start_indices.append(i); regime_types.append("Bearish")
    signals.iloc[i] = 1 if current_regime == "Bullish" else 0

# Extract Current Values for Marking
last_date = sp500.index[-1]
curr_price = sp500.iloc[-1]
curr_ma21 = ma21.iloc[-1]
curr_ma63 = ma63.iloc[-1]
curr_ratio = ratio.iloc[-1]

# 4. Visualization
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 14), sharex=True, 
                                     gridspec_kw={'height_ratios': [2, 1, 1]})

# --- TOP CLIP: S&P 500 INDEX ---
ax1.plot(sp500.index, sp500, color='black', lw=1, label='S&P 500 Index')
ax1.fill_between(sp500.index, ax1.get_ylim()[0], ax1.get_ylim()[1], 
                 where=(signals == 1), color='gray', alpha=0.2)
# Mark Current Price
ax1.scatter(last_date, curr_price, color='black', s=50, zorder=5)
ax1.annotate(f'{curr_price:.2f}', xy=(last_date, curr_price), xytext=(10, 0), 
             textcoords='offset points', fontweight='bold', fontsize=10)
ax1.set_title("S&P 500 INDEX: CURRENT REGIME ANALYSIS", fontweight='bold')
ax1.set_yscale('log')

# --- MIDDLE CLIP: MOVING AVERAGES ---
ax2.plot(ma21.index, ma21, color='blue', label='21-Day MA', lw=1.2)
ax2.plot(ma63.index, ma63, color='red', label='63-Day MA', lw=1.2)
# Mark Current MAs
ax2.scatter([last_date, last_date], [curr_ma21, curr_ma63], color=['blue', 'red'], s=40, zorder=5)
ax2.annotate(f'MA21: {curr_ma21:.2f}', xy=(last_date, curr_ma21), xytext=(10, 5), textcoords='offset points', color='blue')
ax2.annotate(f'MA63: {curr_ma63:.2f}', xy=(last_date, curr_ma63), xytext=(10, -12), textcoords='offset points', color='red')
ax2.set_title("Trend Smoothing")
ax2.legend(loc='upper left')

# --- BOTTOM CLIP: DEVIATION-FROM-TREND RATIO ---
ax3.plot(ratio.index, ratio, color='purple', label='Ratio (21/63 MA)', lw=1.5)
ax3.axhline(1.0, color='black', linestyle='--', alpha=0.3)
# Mark Current Ratio
ax3.scatter(last_date, curr_ratio, color='purple', s=50, zorder=5)
ax3.axhline(curr_ratio, color='purple', linestyle=':', alpha=0.4) # Horizontal line to Y-axis
ax3.annotate(f'Current Ratio: {curr_ratio:.4f}', xy=(last_date, curr_ratio), xytext=(10, 0), 
             textcoords='offset points', fontweight='bold', color='purple')
ax3.set_title("Deviation-From-Trend Ratio")

# Add Current Status Mode Box (Bottom Left)
latest_ret = ((curr_price / sp500.iloc[regime_start_indices[-1]]) - 1) * 100
status_text = (f"CURRENT STATUS: {regime_types[-1].upper()}\n"
               f"Current Ratio: {curr_ratio:.4f}\n"
               f"Regime Return: {latest_ret:+.2f}%")
ax3.text(0.02, 0.15, status_text, transform=ax3.transAxes, fontsize=10,
         bbox=dict(facecolor='white', edgecolor='purple', boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.show()