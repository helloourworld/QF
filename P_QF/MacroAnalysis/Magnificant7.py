# -*- coding: utf-8 -*-
"""
@author: lyu
"""
import pandas as pd
import yfinance as yf
import numpy as np
from Morningstar_to_Access import md, get_morningstar_data
from sqlalchemy import create_engine, text
import urllib
import re

import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.4}'.format


def Mag7_plot(date_start, date_end):
    # Define the ticker symbols of the Magnificent Seven stocks
    tickers = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'TSLA', 'NVDA']
    secid = ['0P000000GY',	'0P000003MH',	'0P000000B7',
             '0P000002HD',	'0P0000W3KZ',	'0P0000OQN8',	'0P000003RE',]
    # Add the S&P 500
    tickers.append('^GSPC')
    tickers.append('^SPXEW')

    secid.append('XIUSA04G92;XI')
    secid.append('XIUSA04ACF;XI')
    # Download historical data as pandas DataFrame
    data = yf.download(tickers, start=date_start, end=date_end)

    # Portfolio Data; data_set_id='7' means get daily return index
    df = md.direct.get_investment_data(secid[:], data_points=[
        {"datapointId": "OS01W"},  # Name
        {
            "datapointId": "HS793",  # Daily Return
            "isTsdp": True,
            "startDate": date_start,
            "endDate": date_end,
        },
    ],).T

    # df = md.direct.portfolio.get_data(portfolio_id=secid[0], data_set_id='7', start_date=str(
    #     date_start), end_date=date_end).T
    df.columns = df.iloc[1, :]
    df = df.iloc[2:, :]
    df.index = pd.Index(list(map(lambda y: y[0], list(
        map(lambda x: re.findall("\d{4}-\d{2}-\d{2}", x), df.index)))))

    # Use only Close price for each stock
    data = data['Adj Close']

    # Return
    data = data.pct_change()
    data = df.pct_change()
    # Calculate the S&P 500 minus the Magnificent Seven
    # mean -> to represent equal weighted
    data.rename(columns={'S&P 500 Equal Weighted TR USD': "^SPXEW",
                         'S&P 500 TR USD': "^GSPC"
                         }, inplace=True)

    data['Eq.S&P 493'] = (data['^SPXEW']*500 -
                          data.iloc[:, :-2].sum(axis=1))/497

    data.index.name = ""
    data.index = pd.to_datetime(data.index)

    # Calculate the correlation matrix
    corr_matrix = data.corr()

    # Print the correlation matrix
    # print(corr_matrix)

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

    # Show the plot
    plt.title("From "+date_start + " To "+date_end)
    plt.show()

    # 493 vs 7
    data['Eq.7 Stocks'] = data.iloc[:, :-2].mean(axis=1)

    # Create a DataFrame with a single row
    new_row = pd.DataFrame([0]*data.shape[1]).T
    new_row.columns = data.columns  # Set the column names to be the same as Size
    data = pd.concat([new_row, data])  # Concatenate the new row with Size
    # Set the index of the first row to "2022-12-31"
    data.index = [pd.to_datetime(date_start) +
                  pd.to_timedelta(-1)] + list(data.index[1:])
    data['Reference'] = 0

    data[['Eq.7 Stocks', 'Eq.S&P 493']].plot(figsize=(12, 9), title="Return")

    # Cumsum
    data_cum = ((1+data/1).cumprod()-1)*1

    data_cum[['Eq.7 Stocks', 'Eq.S&P 493', '^SPXEW', '^GSPC']].plot(
        figsize=(12, 9), title="Cumulative Return")

    # Add labels for the last data point of each column
    for column in ['Eq.7 Stocks', 'Eq.S&P 493', '^GSPC', '^SPXEW', ]:
        haa = 'center'
        offset = -1

        if column == 'Eq.S&P 493':
            offset = -15
        plt.text(data_cum.index[-6], data_cum[column].iloc[offset],
                 f'{column}:{data_cum[column].iloc[-1]: .1%}', ha=haa)


start_dates = ['2018-01-01', '2021-01-01',
               '2022-01-01', '2023-01-01', '2022-01-01',]
end_dates = ['2018-12-31', '2021-12-31',
             '2022-12-31', '2023-11-20', '2023-11-20']

# Pair each start date with its corresponding end date
date_pairs = list(zip(start_dates, end_dates))

# Unpack the pairs and apply Mag7_plot to each pair
results = list(map(Mag7_plot, *zip(*date_pairs)))
