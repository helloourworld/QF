import pandas as pd
import matplotlib.pyplot as plt

# Sample data: Replace with your actual data
data = {'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
        'close': pd.Series(range(100))}

df = pd.DataFrame(data)

# Calculate the moving averages
df['MA5'] = df['close'].rolling(window=5).mean()
df['MA10'] = df['close'].rolling(window=10).mean()

# Calculate the Parabolic SAR
def calculate_sar(df, af=0.02, max_af=0.2):
    high = df['close']
    low = df['close']
    sar = [low[0]]
    trend = 1  # 1 for uptrend, -1 for downtrend
    ep = high[0]  # Extreme point
    af_value = af

    for i in range(1, len(df)):
        if trend == 1:
            sar.append(sar[-1] + af_value * (ep - sar[-1]))
            if low[i] < sar[-1]:
                trend = -1
                sar[-1] = ep
                ep = low[i]
                af_value = af
            else:
                if high[i] > ep:
                    ep = high[i]
                    af_value = min(af_value + af, max_af)
        else:
            sar.append(sar[-1] + af_value * (ep - sar[-1]))
            if high[i] > sar[-1]:
                trend = 1
                sar[-1] = ep
                ep = high[i]
                af_value = af
            else:
                if low[i] < ep:
                    ep = low[i]
                    af_value = min(af_value + af, max_af)

    return sar

df['SAR'] = calculate_sar(df)

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(df['date'], df['close'], label='Close Price')
plt.plot(df['date'], df['MA5'], label='5-Period MA', linewidth=1, color='yellow')
plt.plot(df['date'], df['MA10'], label='10-Period MA', linewidth=1, color='blue')

# Plot Long and Short signals
long_signals = df[df['close'] >= df['SAR']]
short_signals = df[df['close'] <= df['SAR']]
plt.scatter(long_signals['date'], long_signals['SAR'], color='red', label='Long Signal', marker='o')
plt.scatter(short_signals['date'], short_signals['SAR'], color='green', label='Short Signal', marker='o')

plt.legend()
plt.title('Moving Averages and Parabolic SAR')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
