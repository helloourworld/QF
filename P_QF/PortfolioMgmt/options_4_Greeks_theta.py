# https://www.pyquantnews.com/the-pyquant-newsletter/options-lose-value-every-day-measure-theta-decay?utm_source=linkedin&utm_medium=post&utm_campaign=8.26.24.8.LI

import QuantLib as ql
import matplotlib.pyplot as plt

# Define option parameters
expiry_date = ql.Date(15, 6, 2023)
strike_price = 100
spot_price = 105
volatility = 0.2
risk_free_rate = 0.01
dividend_yield = 0.02

# Set up the QuantLib calendar and day count convention
calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
day_count = ql.Actual365Fixed()

# Create the QuantLib objects for the option
exercise = ql.EuropeanExercise(expiry_date)
payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)
option = ql.VanillaOption(payoff, exercise)

# Create the interest rate curve
risk_free_rate_handle = ql.YieldTermStructureHandle(ql.FlatForward(
    0, calendar, ql.QuoteHandle(ql.SimpleQuote(risk_free_rate)), day_count))

# Create the dividend yield curve
dividend_yield_handle = ql.YieldTermStructureHandle(ql.FlatForward(
    0, calendar, ql.QuoteHandle(ql.SimpleQuote(dividend_yield)), day_count))

# Create the volatility surface
volatility_handle = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(
    0, calendar, ql.QuoteHandle(ql.SimpleQuote(volatility)), day_count))

# Create the Black-Scholes process
spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
bs_process = ql.BlackScholesMertonProcess(
    spot_handle, dividend_yield_handle, risk_free_rate_handle, volatility_handle)

# Define the range of days to expiration up until 15 days
days_to_expiry = range(365, 15, -1)

# Container for theta values
theta_values = []

# Calculate theta for each day
for days in days_to_expiry:
    expiry_date = calendar.advance(
        ql.Date().todaysDate(), ql.Period(int(days), ql.Days))
    exercise = ql.EuropeanExercise(expiry_date)
    option = ql.VanillaOption(payoff, exercise)

    # Set up the pricing engine
    engine = ql.AnalyticEuropeanEngine(bs_process)
    option.setPricingEngine(engine)

    # Calculate theta
    theta = option.theta() / 365
    theta_values.append(theta)


# Plot the theta values
plt.figure(figsize=(10, 6))
plt.plot(days_to_expiry, theta_values, label='Theta')
plt.xlabel('Days to Expiration')
plt.ylabel('Theta')
plt.title('Option Theta over Time to Expiration')
plt.gca().invert_xaxis()
ticks = range(365, 15, -50)
plt.xticks(ticks)
plt.show()
