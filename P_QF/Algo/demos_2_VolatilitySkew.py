# -*- coding: utf-8 -*-
"""
@author: lyu

Volatiltiy Skew, Term Structure
"""
import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import matplotlib.pyplot as plt

def fetch_option_data(ticker):
    """
    Fetch option chain data for a given stock ticker.
    """
    Ticker = yf.Ticker(ticker)
    expirations = Ticker.options # Get a list of available expiration dates for options contracts
    option_data = pd.DataFrame()
    
    for expiration in expirations:
        # option chain data
        option_chain = Ticker.option_chain(expiration)

        calls = option_chain.calls
        calls['Option Type'] = "Call"

        puts = option_chain.puts
        puts['Option Type'] = "Put"

        options = pd.concat([calls, puts])

        # actual expiration date by adding 23 hours, 59 minutes, and 59 seconds to the expiration date 
        # (to account for market closing times).
        options['Expiration Date'] = pd.to_datetime(expiration) + pd.DateOffset(hours=23, minutes=59, seconds=59)
        
        option_data = pd.concat([option_data, options])

    # difference between the expiration date and the current date and time.
    option_data["Days to Expiration"] = (option_data['Expiration Date'] - dt.datetime.today()).dt.days + 1
    
    return option_data

if __name__ == "__main__":
    # Fetch option data for NASDAQ
    options_data = fetch_option_data("^NDX")

    # Select call options
    call_options = options_data[options_data["Option Type"] == "Call"]

    # Print available expiration dates
    all_available_expiration_dates = sorted(set(call_options['Expiration Date']))
    print("Available Expiration Dates:")
    print(all_available_expiration_dates)

    # Implied Volatility Skew Plot
    def IV_Skew_plot(chosen_expiry_date):
        # Select an expiration date to plot
        # chosen_expiry_date = "2024-01-31 23:59:59"
        selected_calls_at_expiry = call_options[call_options["Expiration Date"] == chosen_expiry_date]

        # Filter out low implied volatility options
        filtered_calls_at_expiry = selected_calls_at_expiry[selected_calls_at_expiry["impliedVolatility"] >= 0.005]
        
        # Set the strike price as the index for better plotting
        filtered_calls_at_expiry.set_index("strike", inplace=True)
        if filtered_calls_at_expiry.shape[0]>=3:
            # Plot Implied Volatility Skew
            plt.figure(figsize=(10, 6))
            plt.plot(filtered_calls_at_expiry.index, filtered_calls_at_expiry["impliedVolatility"], marker='o', linestyle='-')
            plt.title(f"Implied Volatility Skew for {chosen_expiry_date}")
            plt.xlabel("Strike Price")
            plt.ylabel("Implied Volatility")
            plt.grid(True)
            plt.show()

    for _dt in all_available_expiration_dates:
        IV_Skew_plot(_dt)

    
    # Select a specific strike price to plot
    all_available_strike_price = sorted(set(call_options['strike']))
    print("Available Strike:")
    print(all_available_strike_price)

    # Plot the Implied Volatility Term Structure for the selected strike price. 
    # This chart shows how implied volatility changes across different
    # expiration dates for a specific strike price.
    def Implied_Volatility_Term_Structure(selected_strike_price):
        selected_calls_at_strike = call_options[call_options["strike"] == selected_strike_price]
        # Filter out low implied volatility options
        filtered_calls_at_strike = selected_calls_at_strike[selected_calls_at_strike["impliedVolatility"] >= 0.001]
        # Set the expiration date as the index for better plotting
        filtered_calls_at_strike.set_index("Expiration Date", inplace=True)
        if filtered_calls_at_strike.shape[0]>=15:
            # Plot Implied Volatility Term Structure
            plt.figure(figsize=(12, 10))
            plt.plot(filtered_calls_at_strike.index, filtered_calls_at_strike["impliedVolatility"], marker='o', linestyle='-')
            plt.title(f"Implied Volatility Term Structure for Strike Price {selected_strike_price}")
            plt.xlabel("Expiration Date")
            plt.ylabel("Implied Volatility")
            plt.grid(True)
            plt.show()

    for _strike in all_available_strike_price:
        Implied_Volatility_Term_Structure(_strike)