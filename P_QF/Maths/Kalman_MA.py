import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
"""
Trading teams use it to smooth price-based indicators before making decisions.

A raw moving average of a stock’s price can whip back and forth on volatile days, producing false signals.

The Kalman filter dampens that noise by continuously adjusting how much it trusts each new price relative to its running estimate.

Because it doesn’t need labeled training data or a pre-set window, it reduces the risk of overfitting, which is when a model performs well on past data but fails on new data.

Every time you compute a rolling statistic (like a moving average or a rolling volatility measure), you have to choose how many days to include.

That choice affects your results more than most beginners realize, and there’s no obviously “correct” answer.

The Kalman filter sidesteps this problem by adapting its smoothing at each time step.
"""

# 1. Download Data
ticker = "MCD"
data = yf.download(ticker, start="2026-01-01", )['Close'] # Kalman filter operates on a single series of observations. Working with closing prices is standard practice since they represent the market’s consensus value at the end of each trading session.

# 2. Setup Kalman Filter
# We assume the "true price" is the state we want to find.

"""The key ratio is observation_covariance to transition_covariance.

A small transition_covariance (0.01) relative to observation_covariance (1) tells the filter that most tick-to-tick variation is noise, so it should smooth aggressively."""

kf = KalmanFilter(
    transition_matrices=[1], # transition matrix of 1 tells the filter that tomorrow’s true price is expected to equal today’s (a random walk assumption)
    observation_matrices=[1],
    initial_state_mean=data.values[0],
    initial_state_covariance=1,
    observation_covariance=1,
    transition_covariance=0.01  # This controls how "reactive" the filter is
)

# 3. Apply the filter to the data
state_means, _ = kf.filter(data.values)
kalman_data = pd.Series(state_means.flatten(), index=data.index)

# 4. Calculate Moving Averages for Comparison
sma_20 = data.rolling(window=20).mean()
ema_20 = data.ewm(span=20, adjust=False).mean()

# 5. Plotting the Results
"""In the full view, notice how the Kalman line hugs the price more closely during steady trends and stays smooth during choppy periods. The 30-day moving average, by contrast, lags behind sharp moves because it weights all 30 days equally regardless of market conditions.
The Kalman filter reacts faster because it reweights new information at every step rather than averaging a fixed block of history.
"""
plt.figure(figsize=(14, 7))
plt.plot(data, label='Actual Price (SPY)', color='lightgray', alpha=0.5)
plt.plot(kalman_data, label='Kalman Filter', color='blue', lw=2)
plt.plot(sma_20, label='20-Day SMA', color='red', linestyle='--')
plt.plot(ema_20, label='20-Day EMA', color='green', linestyle='--')

plt.title(f"Kalman Filter vs Moving Averages ({ticker})", fontsize=16)
plt.legend()
plt.show()