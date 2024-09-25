# https://www.pyquantnews.com/the-pyquant-newsletter/poisson-jumps-better-pricing-more-realistic-simulation?utm_source=linkedin&utm_medium=post&utm_campaign=8.29.24.12.LI
# Poisson Jumps: Better pricing with a more realistic simulation
import QuantLib as ql
import matplotlib.pyplot as plt

# Define option parameters
import numpy as np
import matplotlib.pyplot as plt


# Define parameters
S0 = 100  # Initial stock price
mu = 0.05  # Drift
sigma = 0.2  # Volatility
lambda_ = 0.1  # Jump intensity (average number of jumps per year)
mu_j = -0.2  # Mean of the jump size
sigma_j = 0.1  # Standard deviation of the jump size
T = 1  # Time horizon (1 year)
dt = 1/252  # Time step (1 trading day)
N = int(T / dt)  # Number of steps

# Seed for reproducibility
np.random.seed(42)

# Initialize price array
prices = np.zeros(N)
prices[0] = S0

# Simulate the asset price path
for t in range(1, N):
    # Generate the random components
    Z = np.random.normal(0, 1)
    J = np.random.normal(mu_j, sigma_j) if np.random.poisson(
        lambda_ * dt) > 0 else 0

    # Calculate the price
    prices[t] = prices[t-1] * \
        np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z + J)


# Plot the simulated price path
plt.figure(figsize=(10, 6))
plt.plot(prices, label='Simulated Asset Price with Jumps')
plt.xlabel('Time (days)')
plt.ylabel('Price')
plt.title('Simulated Asset Price Path with Poisson Jumps')
plt.legend()
plt.show()
