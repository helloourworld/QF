import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=2, suppress=True)

# 1. SETUP SYNTHETIC DATA
np.random.seed(42)
n_days = 50
true_price = 180 + np.cumsum(np.random.normal(0, 1, n_days)) # The "Real" Trend
observed_price = true_price + np.random.normal(0, 3, n_days) # The "Noisy" Ticker
print("True Price (Hidden):", true_price[:10])  # Show first 10 for brevity
print("Observed Price (Noisy):", observed_price[:10])  # Show first 10 for brevity
# 2. INITIALIZE KALMAN PARAMETERS
"""
The Prediction Error (P)
If you stop looking at the market for 10 days, your P grows. You become less certain about where the price is. The Kalman filter mathematically models this: P = P + Q.
 
P = Estimate of how certain we are about our current guess (lower is more certain)
Q = How much we expect the true price to change day-to-day (process noise)

Kalman Gain (K)
K = P / (P + R)
If Market Noise R is high: K becomes a very small decimal (like 0.05). The filter says: "The ticker is lying, don't move my estimate much."
If our Certainty P is low: K becomes larger. The filter says: "I don't know what's happening, I better follow the ticker closely."
 
R = How noisy we think the ticker is (measurement noise)

The State Update:
New Estimate = Old Estimate + K * (Difference between Ticker and Estimate)
This is identical to a Weighted Moving Average, but K is not fixed (like 0.1). K re-calculates itself every single day based on how much the stock is vibrating.
"""

# Starting guess
x = observed_price[0] 
P = 1.0               

# Hyper-parameters (The "Tuning Knobs")
Q = 0.1   # Process Noise: We assume the true trend is stable
R = 9.0   # Measurement Noise: We assume the ticker is very **jumpy** (Standard Dev squared)

kalman_estimates = []

print(f"{'Day':<5} | {'Ticker':<10} | {'Kalman Guess':<12} | {'Gain (K)':<10} | {'Certainty (P)':<10}")
print("-" * 60)

# 3. THE KALMAN LOOP
for i in range(n_days):
    z = observed_price[i] # Current Ticker Measurement
    
    # --- STEP A: PREDICT ---
    # We predict the price today will be the same as yesterday
    
    # But our uncertainty (P) grows because time has passed (Q)
    x_pred = x
    P_pred = P + Q
    
    # --- STEP B: UPDATE (The Math Details) ---
    # 1. Calculate Kalman Gain (K)
    # K tells us how much to weight the new measurement z.
    # If R (noise) is huge, K becomes small (we ignore the ticker).
    K = P_pred / (P_pred + R)
    
    # 2. Update the Estimate (x)
    # New Estimate = Old Estimate + K * (Difference between Ticker and Estimate)
    x = x_pred + K * (z - x_pred)
    
    # 3. Update the Certainty (P)
    # We become more certain because we have more data points.
    P = (1 - K) * P_pred
    
    kalman_estimates.append(x)
    
    if i % 3 == 0:
        print(f"{i:<5} | {z:<10.2f} | {x:<12.2f} | {K:<10.4f} | {P:<10.4f}")

# 4. VISUALIZATION
plt.figure(figsize=(12, 6))
plt.plot(true_price, label='True Signal (Hidden)', color='green', marker='^', alpha=0.5, ls='--')
plt.plot(observed_price, label='Noisy Ticker (Actual Data)', color='red', marker='o', alpha=0.3)
plt.plot(kalman_estimates, label='Kalman Filter Estimate', color='blue', lw=2)
plt.title("Kalman Filter: Denoising NVDA Price Action")
plt.legend()
plt.show()