import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import timedelta, datetime
import time
# import datatable as dt
import pandas as pd
import numpy as np
import os


def shrink_to_diagonal(sigma, shrinkage, shrink_to_unit=False):
    if not shrink_to_unit:
        return (1 - shrinkage) * sigma + shrinkage * np.diag(np.diag(sigma))
    else:
        if np.sum(np.abs(sigma)) > 0:
            sigma = sigma / np.sum(np.abs(sigma))
        return (1 - shrinkage) * sigma + shrinkage * np.diag(1 + 0 * np.diag(sigma))


def reduce_dimension_of_signals_and_returns(returns, signal_list, reduced_dimension, rolling_window=120):
    """
    pick top PCs of returns and then reduce dimensions of signals and returns
    :param returns:
    :param signal_list:
    :param reduced_dimension:
    :param rolling_window:
    :return:
    """

    transformed_r = returns.copy().fillna(0).iloc[:, -reduced_dimension:] * 0
    # tmp = transformed_r.iloc[:, :2].copy() * 0
    # populate zeros
    transformed_signals = list()
    for jj in range(len(signal_list)):
        signal_list[jj] = signal_list[jj].fillna(0)
        transformed_signals += [transformed_r.copy() * 0]

    previous_vectors = np.array([0])

    for ii in range(rolling_window + 1, returns.shape[0]):
        r_slice = returns.iloc[(ii - rolling_window):(ii - 1 + 1), :].fillna(0).values
        sigma = np.matmul(r_slice.T, r_slice)
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)
        if np.sum(eigenvalues > 0.0001) < reduced_dimension:
            continue
        if previous_vectors.shape[0] == 1 and np.sum(eigenvalues) > 0:
            previous_vectors = eigenvectors
        else:
            sign_flips = np.sign(np.diag(np.matmul(previous_vectors.T, eigenvectors)))
            # print('flips', ii, np.sum(sign_flips < 0))
            eigenvectors = np.matmul(eigenvectors, np.diag(sign_flips))
            # print(pd.DataFrame(np.append(eigenvectors[:, sign_flips < 0][:, -2:], previous_vectors[:, sign_flips < 0][:, -2:], axis=1)))
            previous_vectors = eigenvectors
        v_vec = eigenvectors[:, -reduced_dimension:]
        r = returns.iloc[ii, :].values.reshape(-1, 1)
        transformed_r.iloc[ii, :] = np.matmul(v_vec.T, r).flatten()
        r1 = transformed_r.iloc[ii, :]

        for jj in range(len(signal_list)):
            s = signal_list[jj].iloc[ii, :].values.reshape(-1, 1)
            transformed_signals[jj].iloc[ii, :] = np.matmul(v_vec.T, s).flatten()
            s1 = transformed_signals[jj].iloc[ii, :]
        # tmp.iloc[ii, 0] = (r * s).sum()
        # tmp.iloc[ii, 1] = (r1 * s1).sum()

    for jj in range(len(signal_list)):
        transformed_signals[jj] = transformed_signals[jj].iloc[rolling_window:, :]
    return transformed_r.iloc[rolling_window:, :], transformed_signals #, tmp


def rank_dataframe(data_frame, good_columns, fillnans=True, axis=0):
    """
    ranks signals
    :param data_frame:
    :param good_columns: columns to be ranked
    :param fillnans: whether we fill Na with zeros
    :return:
    """
    data = data_frame.copy()
    all_dates = data.DATE.unique()
    data.index = data.DATE
    for date in all_dates:
        if fillnans:
            val = data.loc[date, good_columns].rank(pct=True, axis=axis).fillna(0.5) - 0.5
        else:
            val = data.loc[date, good_columns].rank(pct=True, axis=axis) - 0.5
        data.loc[date, good_columns] = val
    return data


def vol_adjust_data(rets, periods=24, monthly=True):
    """
    simple vol adjusted returns
    :param rets:
    :param periods:
    :return:
    """
    if monthly:
        ss = 1
    else:
        ss = 2
    tmp = rets.rolling(periods).std(min_periods=int(np.round(0.8 * periods))).shift(ss)
    tmp[tmp == 0] = np.nan
    tmp1 = rets / tmp
    return tmp1


def add_last_column_desired(portfolio_returns, columns_list, new_name):
    tmp = best_convex_combination_of_returns(portfolio_returns[columns_list])
    portfolio_returns[new_name] = tmp
    return portfolio_returns


def efficient_portfolio(returns_frame: pd.DataFrame):
    returns_frame = returns_frame.astype(float)
    mu = returns_frame.mean(0).values
    sigma = returns_frame.cov().values
    sig_inv = quasi_inverse(sigma, 0.001)
    pi = np.matmul(sig_inv, mu.reshape(-1, 1))
    ut = np.matmul(mu, np.matmul(sig_inv, mu.T))
    return pi, ut


def best_convex_combination_of_returns(returns_frame):
    pi1, _ = efficient_portfolio(returns_frame)
    if np.min(pi1) >= 0:
        return np.matmul(returns_frame.values, pi1)
    else:
        if pi1.shape[0] == 2:
            return np.matmul(returns_frame.values, 0 * pi1)
        if pi1.shape[0] == 3:
            best_u = 0
            best_pi = np.zeros([2, 1])
            best_ind = [0, 1]
            for ind in [[0, 1], [0, 2], [1, 2]]:
                pi1, ut = efficient_portfolio(returns_frame.iloc[:, ind])
                if ut > best_u and np.min(pi1) >= 0:
                    best_u = ut
                    best_pi = pi1
                    best_ind = ind
            return np.matmul(returns_frame.iloc[:, best_ind].values, best_pi)


def compute_and_save_number_of_months_per_stock():
    data = pd.read_csv('GHZ_ZHY_V8.csv')
    how_many = data['DATE'].groupby(data.permno).count()
    how_many.to_csv('number_month_per_stock.csv')


def read_ff_data(file_name, directory):
    """
    read Fama French data (any universe)
    :param file_name:
    :param directory:
    :return:
    """
    if file_name == '100_Size_BM.csv':
        path = os.path.join('.', 'FamaFrench', 'data', '100_Size_BM.csv')
        returns = pd.read_csv(path).astype(float)
        path = os.path.join('.', 'FamaFrench', 'data', '100_Size_BM_dates.csv')
        ind = pd.read_csv(path).astype(int).values
        returns.index = [int(k) for k in ind]
        returns.index = [datetime.strptime(str(k), '%Y%m%d') for k in returns.index]
        returns[returns < - 99] = np.nan
    else:
        returns = pd.read_csv(os.path.join(directory, file_name))
        if returns.iloc[0, 0] < 10**6:
            returns.iloc[:, 0] = returns.iloc[:, 0] * 100 + 28
        returns.index = [datetime.strptime(str(k), '%Y%m%d') for k in returns.iloc[:, 0]]
        returns = returns.iloc[:, 1:]
        returns[returns < - 99] = np.nan
        if 'MarketCap' in file_name:
            returns = -1 + returns / returns.shift(12)
    return returns


def re_arrange_slice(current_slice, ranking=False):
    start_t1 = time.time()

    stock_list = np.unique(current_slice.index)
    print(f'found {len(stock_list)} stocks')
    # local_function = partial(extract, current_slice, stock_list)

    # first we keep only stocks that have all dates
    how_many = current_slice.iloc[:, 0].groupby(current_slice.index).count()
    full_stocks = how_many == max(how_many)
    current_slice = current_slice.loc[full_stocks.loc[full_stocks].index].sort_index()

    available_dates = np.unique(current_slice['DATE'])
    non_stocks = set(current_slice.columns) - set(['RET', 'DATE'])
    if ranking:
        for date in available_dates:
            index = current_slice.DATE == date
            current_slice.loc[index, non_stocks] \
                = current_slice.loc[index, non_stocks].rank(pct=True, axis=1).fillna(0.5) - 0.5
    else:
        for date in available_dates:
            index = current_slice.DATE == date
            current_slice.loc[index, non_stocks] \
                = current_slice.loc[index, non_stocks] - current_slice.loc[index, non_stocks].mean(
                0).values.reshape(1, -1)

    sorted_stocks = np.unique(current_slice.index)
    stock_labels = [f'{stock}_RET' for stock in sorted_stocks]
    the_big_column_set = list()

    signal_labels = dict()
    for col in current_slice.columns[1:]:
        signal_labels[col] = [f'{stock}_{col}' for stock in sorted_stocks]

    for stock in sorted_stocks:
        # warning, I am dropping DATE
        the_big_column_set = the_big_column_set + [f'{stock}_{col}' for col in current_slice.columns[1:]]

    available_dates = current_slice.DATE.unique()
    total_true_data = np.zeros([len(available_dates), len(the_big_column_set)])# pd.DataFrame(columns=the_big_column_set)
    np_slice = current_slice.values[:, 1:]
    for index, date in enumerate(available_dates):
        on_the_date = np_slice[current_slice.DATE == date, :]
        on_the_date = on_the_date.flatten()
        total_true_data[index, :] = on_the_date

    total_true_data = pd.DataFrame(total_true_data, index=available_dates, columns=the_big_column_set)
    end_t = time.time()
    print(f' re-arranging data for {[available_dates[0], available_dates[-1]]} in total took {end_t - start_t1} seconds')

    return total_true_data, stock_labels, signal_labels


def demean_cross_section(data_frame: pd.DataFrame, standardize=False):
    # use numpy broadcasting for a truly Pythonic demeaning
    if not standardize:
        if len(data_frame.shape) > 1:
            return data_frame - data_frame.mean(axis=1).values.reshape(-1, 1)
        else:
            return data_frame - data_frame.mean()
    else:
        if len(data_frame.shape) > 1:
            return (data_frame - data_frame.mean(axis=1).values.reshape(-1, 1)) / data_frame.std(axis=1).values.reshape(-1, 1)
        else:
            return (data_frame - data_frame.mean()) / data_frame.std()


def lift_to_missing_data(r_slice, good_indices, vectors):
    full_relevant_vectors = np.zeros([r_slice.shape[1], vectors.shape[1]])
    full_relevant_vectors[good_indices, :] = vectors
    return full_relevant_vectors


def matrix_inverse_square_root(matrix_a, cond):
    w, v = np.linalg.eigh(matrix_a)
    w = np.sqrt(np.abs(w))
    w = w.flatten().clip(cond * max(w))
    b = np.matmul(v, np.matmul(np.diag(1 / w), v.T))
    return b


def orthogonalize(time_series, cond=0.0001):
    if isinstance(time_series, pd.DataFrame):
        time_series = time_series.values
    matrix = np.matmul(time_series, time_series.T)
    normalizer = matrix_inverse_square_root(matrix, cond)
    return np.matmul(normalizer, time_series)


def matrix_square_root(matrix_a):
    """
    :param matrix_a: positive semi'definite matrix
    :return: matrix square root of A
    """
    w, v = np.linalg.eigh(matrix_a)  # important to use eigh; eig has a bug
    w = np.sqrt(np.abs(w))
    b = np.matmul(v, np.matmul(np.diag(w), v.T))
    return b


def quasi_inverse(matrix_a, cond=0.001):
    """
    :param matrix_a: positive semi-definite matrix A
    :param cond: conditioning number in (0,1): we will clip einvalues so that lambda_min >= cond * lambda_max
    :return: quasi-inverse of A
    """
    w, v = np.linalg.eigh(matrix_a)
    if max(w) < 0.0000000001:
        return 0 * v
    w = np.abs(w).flatten().clip(cond * max(w))
    b = np.matmul(v, np.matmul(np.diag(1 / w), v.T))
    return b


def reindex_monthly_to_date(list_of_dates, monthly_date):
    for dat in list_of_dates:
        if monthly_date.year == dat.year and monthly_date.month == dat.month:
            return dat
    return monthly_date.date()


# def make_pd(df: dt.Frame):
#     tmp = df.to_pandas()
#     tmp.index = tmp.DATE
#     tmp = tmp.iloc[:, 1:]
#     return tmp


def quadratic_form(matrix, vector):
    if len(vector.shape) == 1 or vector.shape[1] > 1:
        vector = vector.reshape(-1, 1)
    x_ = np.matmul(vector.T, matrix)
    return np.matmul(x_, vector)


def sharpe_ratio(returns, monthly=False):
    """
    :param returns: returns: pandas DataFrame
    :return: annualized sharpe Ratio
    """
    if not isinstance(returns, pd.DataFrame) and not isinstance(returns, pd.Series):
        raise Exception('Returns must be a pandas dataframe')
    if monthly:
        mult = 12
    else:
        mult = 256
    sh = np.sqrt(mult) * returns.mean(0) / returns.std(0)
    return np.round(sh, 2)


def plot_portfolio_returns(portfolio_returns: pd.DataFrame, ff_factors: pd.DataFrame, address: str, monthly=False,
                           control_factor=True, show_alpha=False, just_market=False):
    """
    Key
    :param portfolio_returns: portfolio returns
    :param ff_factors: ff factors for alpha computation
    :param address: saving folder
    :param monthly: True if monthly returns
    :param control_factor: if true, then we should incude factor into the right-hand side for alpha computation
    :return:
    """
    sharpes, tstat = compute_performance_characteristics(portfolio_returns, ff_factors,
                                                         monthly=monthly, control_factor=control_factor, just_market=just_market)

    subset_col = portfolio_returns.columns
    plt.figure()
    plt.plot(portfolio_returns[subset_col].astype(float).dropna(0).cumsum())
    if show_alpha:
        titl = f'{sharpes.values.flatten()}\n alpha t-stats = {np.round(tstat.values.flatten())}'
    else:
        titl = f'{sharpes.values.flatten()}'
    plt.title(titl)
    plt.legend(portfolio_returns[subset_col].columns)
    plt.savefig(address)
    plt.close('all')
    return sharpes, tstat


def add_last_columns(portfolio_returns):
    tmp = best_convex_combination_of_returns(portfolio_returns[['sym_neg_4', 'pos+neg']])
    portfolio_returns['efficientNEG4+F'] = tmp
    portfolio_returns['efficientNEG4+ASYM4+F'] = best_convex_combination_of_returns(portfolio_returns[['sym_neg_4', 'asym_4', 'pos+neg']])
    portfolio_returns['topposneg'] = portfolio_returns[['sym_pos_4', 'sym_neg_4']].sum(1)
    portfolio_returns['efficientPOSNEG4+ASYM4+F'] = best_convex_combination_of_returns(portfolio_returns[['topposneg', 'asym_4', 'pos+neg']])
    return portfolio_returns


def compute_performance_characteristics(portfolio_returns: pd.DataFrame, ff_factors: pd.DataFrame,
                                        monthly=False, control_factor=True, just_market=False, add_additional_columns=None):
    """
    this function computes sharpe ratios and tstat of alpha from regressing portfolio returns on ff_factors
    :param portfolio_returns: pandas dataframe
    :param ff_factors:
    :return: sharpes, tstat
    """
    portfolio_returns.sort_index(inplace=True)
    portfolio_returns = portfolio_returns.dropna(axis=0).astype(float)

    # now we add efficient combinations
    if 'sym_neg_4' in portfolio_returns.columns:
        add_last_columns(portfolio_returns)
    else:
        if add_additional_columns:
            for qq in add_additional_columns:
                add_last_column_desired(portfolio_returns, qq['column_list'], qq['name'])

    sharpes = np.round(sharpe_ratio(portfolio_returns, monthly=monthly), 1)
    if portfolio_returns.shape[0] < 50:
        return sharpes, sharpes
    col = []
    if 'RF' in ff_factors.columns:
        col = ['RF']
    x_ = ff_factors.reindex(portfolio_returns.index).drop(columns=col).dropna(axis=0)
    if just_market and 'Mkt-RF' in ff_factors.columns:
        x_ = x_['Mkt-RF']
    portfolio_returns = portfolio_returns.reindex(x_.index).astype(float)

    x_ = sm.add_constant(x_)
    if control_factor and 'factor' in portfolio_returns.columns:
        x_['factor'] = portfolio_returns['factor']

    # run multi-variate regression
    tstat = np.zeros([1, portfolio_returns.shape[1]])
    for ii, col in enumerate(portfolio_returns.columns):
        y_ = portfolio_returns[col]
        # Newey-West standard errors with maxlags
        z_ = x_.copy().astype(float)
        if col == 'factor' and control_factor:
            z_.drop(columns='factor', inplace=True)
        result = sm.OLS(y_.values, z_.values).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
        try:
            tstat[0, ii] = np.round(result.summary2().tables[1]['z']['const'], 1)  # alpha t-stat (because for 'const')
        except:
            print(f'something is wrong for t-stats')
    tstat = pd.DataFrame(tstat, columns=portfolio_returns.columns)
    return sharpes, tstat


def regression_with_tstats(predicted_variable, explanatory_variables):
    x_ = explanatory_variables
    x_ = sm.add_constant(x_)
    y_ = predicted_variable
    # Newey-West standard errors with maxlags
    z_ = x_.copy().astype(float)
    result = sm.OLS(y_.values, z_.values).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
    try:
        tstat = np.round(result.summary2().tables[1]['z'], 1)  # alpha t-stat (because for 'const')
        tstat.index = list(z_.columns)
    except:
        print(f'something is wrong for t-stats')
    return tstat


def regression_alphas(portfolio_returns, right_hand_side, add_constant=True):
    if add_constant:
        right_hand_side = sm.add_constant(right_hand_side)
    # run multi-variate regression
    tstat = np.zeros([1, portfolio_returns.shape[1]])
    for ii, col in enumerate(portfolio_returns.columns):
        y_ = portfolio_returns[col]
        # Newey-West standard errors with maxlags
        z_ = right_hand_side.copy()
        result = sm.OLS(y_, z_).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
        tstat[0, ii] = np.round(result.summary2().tables[1]['z']['const'], 1)  # alpha t-stat (because for 'const')
    tstat = pd.DataFrame(tstat, columns=portfolio_returns.columns)
    return tstat

