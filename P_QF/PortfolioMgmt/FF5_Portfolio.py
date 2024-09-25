import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

# Fetch ticker of S&P500 constituents
SP_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP_table = pd.read_html(SP_url)[0]
ticker_symbols = SP_table["Symbol"]
ticker_symbols = ticker_symbols.to_list()
# Fetch Historical data on Yahoo Finance
P = yf.download(ticker_symbols, start=datetime(2019, 1, 1),
                )["Adj Close"]
# Align Excess_Returnber of columns dropping the stock with inconsistent Excess_Returnber of observation
null = P.describe().T.reset_index()
null = null["Ticker"][null["count"] < 1000]
null = null.to_list()
P = P.drop(labels=null, axis=1)

# Upload the FFC Risk Factor Data downloaded by FAMA-FRENCH website

Factors = pd.read_csv("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
                 compression='zip', header=2, sep=',', quotechar='"', parse_dates=['Unnamed: 0'])
Factors.set_index('Unnamed: 0', inplace=True)
# Manipulate Portfolio_Factors and Return calculation
P.index = pd.to_datetime(P.index.strftime("%Y-%m-%d"))
Factors.index = pd.to_datetime(Factors.index)
Portfolio_Factors = pd.merge(
    P, Factors, how="left", left_index=True, right_index=True)

Portfolio_Factors = Portfolio_Factors.dropna(
    subset=['SMB'])  # Drop Nan in Factors
RF = Portfolio_Factors["RF_y"]

Factors = Portfolio_Factors[["Mkt-RF", "HML", "SMB", "CMA", "RMW"]]

R = Portfolio_Factors.drop(
    labels=["Mkt-RF", "HML", "SMB", "RMW", "CMA", "RF_y"], axis=1).pct_change().fillna(0)
ExRet = R.subtract(RF[1:], axis=0)

# Fetch the benchmark data
SP = yf.download("^GSPC"
                 )["Adj Close"]
SPR = SP.pct_change().dropna()

SPR = SPR[R.index]
# Features calculation
sigma_m = SPR.std()*np.sqrt(252)  # Std Dev S&P Returns annualized

sigma_i = pd.DataFrame(R.std()*np.sqrt(252))  # Std Dev any stock annualized
sigma_i.columns = ["sigma_i"]

# Expected Return any stock annualized
Rbar = pd.DataFrame(((1+R.mean())**252)-1)
Rbar.columns = ["R_mean"]

RFm = (RF.mean())

# Single index model estimated for any FFC factor with any stock in S&P pool
universe_weights = {factor: []
                    for factor in Factors.columns}  # dict to store weigths

# dict to store optimal stock list
universe_list = {factor: [] for factor in Factors.columns}


for Factor in Factors.columns:
    betas = []
    X = Factors[1:][Factor]
    X = sm.add_constant(X)

    regress_data = pd.concat([ExRet*100, X], axis=1).dropna()

    col_name = X.columns.to_list()

    for stock in ExRet.columns:
        model = sm.OLS(regress_data[stock], regress_data[col_name])
        fit = model.fit(mcov_type='HAC')
        # print(f'{fit.summary(yname=stock,xname=col_name)}')
        betas.append(fit.params)
    ticker = pd.DataFrame(ExRet.columns)
    ticker.columns = ["Ticker"]
    betas = pd.merge(ticker, pd.DataFrame(betas), how="left",
                     left_index=True, right_index=True)
    betas = betas.set_index("Ticker")

    # Process for select optimal stock with the C* law for single/multi index model
    tab = pd.merge(betas, Rbar, how="left", left_index=True, right_index=True)
    tab = pd.merge(tab, sigma_i, how="left", left_index=True, right_index=True)

    # Excess Return
    tab["Excess_Return"] = tab["R_mean"].subtract(RFm)

    # **Beta ratio**
    tab["ExRet_beta"] = tab["Excess_Return"] / tab[col_name[1]]

    tab = tab.sort_values("ExRet_beta", ascending=False)

    # Calculation of the single components for the final C* calculus
    tab2 = tab.copy()
    tab2["step1"] = (tab2["ExRet_beta"]*tab2[col_name[1]]) / tab2["sigma_i"]**2

    tab2["step2"] = tab2[col_name[1]]**2 / tab2["sigma_i"]**2

    tab2["Excess_Return2"] = tab2["step1"].cumsum()

    tab2["den"] = tab2["step2"].cumsum()

    # Ci Calculus and boolean compare for selecting criteria
    tab2["Ci"] = (sigma_m**2 * tab2["Excess_Return2"]) / \
        (1 + sigma_m**2 * tab2["den"])

    tab2["Optimal"] = tab2["ExRet_beta"] > tab2["Ci"]

    # Export in Excel
    # tab2.to_excel(f"C:\\Users\\cdecinti\\desktop\\report_{Factor}.xlsx")

    # Pesi ottimali
    weights = tab2[tab2["Optimal"] == True]

    Ci = weights["Ci"]  # .iloc[-1]

    weights["Z"] = (weights["ExRet_beta"] - Ci).T * \
        (weights[col_name[1]] / weights["sigma_i"]**2)

    # Weights
    weights["weights"] = weights["Z"] / weights["Z"].sum()

    appo = weights["weights"].reset_index()
    universe_weights[Factor] = appo["weights"]

    universe_list[Factor] = weights.index.to_list()

universe_weights = list(universe_weights.values())
universe_list = list(universe_list.values())


factors = ['MKT', 'HML', 'SMB', 'CMA', 'RMW']
lists = {factor: universe_list[i] for i, factor in enumerate(factors)}
weights = {factor: universe_weights[i] for i, factor in enumerate(factors)}
selected_returns = {factor: R[lists[factor]] for factor in factors}

# Combine all portfolio returns into a single DataFrame
portfolio_returns = pd.DataFrame(
    [np.dot(R[lists[factor]], weights[factor]) for factor in factors]).T


portfolio_returns.index = SPR.index
ret = pd.concat([portfolio_returns, SPR], axis=1)

# Recol_name columns
ret.columns = factors + ['S&P 500 Benchmark']
# Plot cumulative Returns
plt.rcParams["figure.figsize"] = (12, 6)
ret.cumsum().plot(title="Cumulative Returns")

# Downside Risk
Downside = ret.applymap(lambda x: x if x < 0 else 0)

colors = ['green', 'red', 'yellow', 'orange', 'purple', 'blue']
# Define the grid size
n_rows, n_cols = 3, 2

# Create a figure and axis array
fig, axs = plt.subplots(n_rows, n_cols, figsize=(30, 20))

# Loop through the rows and columns using eExcess_Returnerate
for idx, ax in enumerate(axs.flat):

    ax.plot(Downside[ret.columns[idx]] * 100,
            color=colors[idx], label="Downside")
    ax1a = ax.twinx()
    ax1a.plot(Downside['MKT'].rolling(5).std() * np.sqrt(252) * 100,
              color="darkgrey", label="Downside deviation", alpha=.8)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1a.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Downside Risk and Downside deviations for {} portfolio".format(
        ret.columns[idx]))

plt.show()

# Drawdown Risk
Drawdown = pd.DataFrame()
for Portfolio in ret.columns:
    cum_wealth = (1+ret[[Portfolio]]).cumprod() * \
        100  # -1 # ret[[Portfolio]].cumsum()
    # cum.plot()
    cum_max = cum_wealth.cummax()

    Drawdown[Portfolio] = (cum_wealth-cum_max)/cum_max

# Loop through the rows and columns using eExcess_Returnerate
# Create a figure and axis array
fig, axs = plt.subplots(n_rows, n_cols, figsize=(30, 20))
for idx, ax in enumerate(axs.flat):

    # Use the fill_between method to fill between the lines
    ax.fill_between(
        Drawdown.index, Drawdown[ret.columns[idx]]*100, color=colors[idx], label="Drawdown")

    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1a.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set_title("Drawdown Risk for {} portfolio".format(ret.columns[idx]))

plt.show()
