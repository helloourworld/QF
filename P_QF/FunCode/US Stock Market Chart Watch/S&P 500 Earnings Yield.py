import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# https://www.ndr.com/explain/S672
# 1. Helper function for Front-Weighted Moving Average (WMA)
def front_weighted_ma(series, window):
    weights = np.arange(1, window + 1)
    return series.rolling(window).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

# 2. Load / Prepare Data
# Note: You should replace this with your own source for S&P 500 Price and Earnings Yield.
# Earnings Yield = Trailing 12-Month Earnings / Index Price.
data = pd.read_excel('./FunCode/US Stock Market Chart Watch/SP500.xlsx', index_col=0, parse_dates=True)  # Uncomment when you have the data
# data = pd.read_csv('sp500_data.csv', index_col='Date', parse_dates=True)
Earnings_data = data.iloc[:,1:]  # Assuming the next columns are earnings data
# Placeholder: Generating synthetic data for demonstration
print(data.head())

data['SP500_Index'] = data.iloc[:, 0]  # Assuming the first column is the S&P 500 Index



Earnings_data.index = pd.to_datetime(Earnings_data['Unnamed: 2'], infer_datetime_format=True)  # Assuming 'Unnamed: 2' is the date column in earnings data
Earnings_data['Earnings_Yield'] = Earnings_data.iloc[:,2]
print(Earnings_data.head())

data = data.join(Earnings_data['Earnings_Yield'], how='inner')  # Join on date index
print(data.head())
# 3. Calculate Indicators
# Middle Clip: 39-week and 52-week Front-Weighted Moving Averages
data['WMA_39'] = front_weighted_ma(data['Earnings_Yield'], round(39/4))
data['WMA_52'] = front_weighted_ma(data['Earnings_Yield'], round(52/4))

# Bottom Clip: Deviation-from-trend ratio
data['Ratio'] = data['WMA_39'] / data['WMA_52']

# Signal Thresholds: 78-week moving average and standard deviation of the Ratio
data['Ratio_MA78'] = data['Ratio'].rolling(78).mean()
data['Ratio_Std78'] = data['Ratio'].rolling(78).std()

data['Upper_Bracket'] = data['Ratio_MA78'] + (0.5 * data['Ratio_Std78'])
data['Lower_Bracket'] = data['Ratio_MA78'] - (2.0 * data['Ratio_Std78'])

# 4. Generate Signals
data['Signal'] = 0  # 1 for Bullish, -1 for Bearish, 0 for Neutral
data.loc[data['Ratio'] > data['Upper_Bracket'], 'Signal'] = 1
data.loc[data['Ratio'] < data['Lower_Bracket'], 'Signal'] = -1
# Carry signal forward until the opposite occurs (simple regime logic)
data['Regime'] = data['Signal'].replace(0, np.nan).ffill().fillna(0)

# 5. Plotting (3 Clips)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

# Top Clip: S&P 500 Index
ax1.plot(data.index, data['SP500_Index'], color='black', label='S&P 500 Index')
ax1.set_title('S&P 500 INDEX VS S&P 500 EARNINGS YIELD')
ax1.legend(loc='upper left')
ax1.set_yscale('log')

# Middle Clip: Smoothed Earnings Yield (FWMAs)
ax2.plot(data.index, data['WMA_39'], label='39-Week FWMA', color='blue', alpha=0.8)
ax2.plot(data.index, data['WMA_52'], label='52-Week FWMA', color='red', alpha=0.8)
ax2.set_title('Smoothed S&P 500 Earnings Yield')
ax2.legend(loc='upper left')

# Bottom Clip: Deviation Ratio and Brackets
ax3.plot(data.index, data['Ratio'], label='Deviation-from-trend Ratio', color='purple')
ax3.plot(data.index, data['Upper_Bracket'], '--', label='Bullish (+0.5 SD)', color='green')
ax3.plot(data.index, data['Lower_Bracket'], '--', label='Bearish (-2.0 SD)', color='red')
ax3.fill_between(data.index, data['Lower_Bracket'], data['Upper_Bracket'], color='grey', alpha=0.1)
ax3.set_title('Deviation-from-Trend Ratio (39-Wk MA / 52-Wk MA)')
ax3.legend(loc='upper left')

# Highlight Signal Regimes
for i in range(1, len(data)):
    if data['Regime'].iloc[i] == 1:
        ax1.axvspan(data.index[i-1], data.index[i], color='green', alpha=0.05)
    elif data['Regime'].iloc[i] == -1:
        ax1.axvspan(data.index[i-1], data.index[i], color='red', alpha=0.05)

plt.tight_layout()
plt.show()