import multiprocessing as mp
import random
import time
from itertools import combinations  # permutations

import numpy as np
import statsmodels.api as sm
from numpy import linalg as LA

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import sys
from plot_functions import *
import os
from simulation_functions import Simulate_OOS_Signals_and_Returns
from expectation_functions import marcenko_pastur, derivative_of_marcenko_pastur
random.seed(10)

# Parameter setting
list_of_z = np.array([1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, \
                      1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5, 1e+6])  # 1e-16, 1e-14,
N = 1
T_TRAIN = 240 # 120  # 45 # 52 #45 # 95  # fixed
T_OOS = 12  # 12 * 10
# STARTING_POINT = 1100 - T_TRAIN - T_OOS - 8 # - 12*i # 1100 - T_TRAIN - T_OOS - 8  # 1128 - T_TRAIN - T_OOS - 2 # Only one piece of data slice
# 1128 - T_TRAIN - T_OOS - 501 # 500 piece of data slice
# 500  # Don't start from 0, because 0 is nan

DEGREE = 4
rho = np.array([0.01, 0.02, 0.03, 0.04])
EIGENVALUE_THRESHOLD = 0  # 1e-10
DISCRETE_SIGNALS_EXPANSION = False
DEMEAN_AND_STANDARDIZE_SIGNALS = True
number_of_combinations = 1 # 1 is used for debug, there is no permutation of signal for number_of_combinations = 1
# 100: --- 18544.280282974243 seconds ---
PERMUTE_EIGENVECTORS = False # True
USE_FOURIER_FEATURES = False # True # False
normalize_returns = True # False

number_of_PC_signals = 1000 # 1000 out of 1470 # 700 # 3600 # out of 4095
# 700
DEMEAN_RETURNS_SEPARATELY_INSAMPLE_OOS = False # This parameter controls whether to demean the returns in-sample and oos separately
M_step = 10

signals_adding_method_index = 1
signals_adding_methods = ['poly order', 'eigval descending', 'eigval ascending', 'random']
signals_adding_method = signals_adding_methods[signals_adding_method_index]
# The four method of adding signals to increase c:
# 'poly order': add signals in the order of PolynomialFeatures output
# 'eigval descending': sort eigenvalues of A_T and add signals by descending eigenvalues
# 'eigval ascending': sort eigenvalues of A_T and add signals by ascending eigenvalues
# 'random': add signals randomly

performance_names = ['R-square', 'SRU', 'Return', 'Risk', 'MSE', 't-stats', 'OOSMeanReturn_SRU', 'OOSMeanReturn_Return',\
                     'OOSMeanReturn_Risk', 'ptf_minus_coef_times_mkt_sharpes', 'beta_2nd_norm']
# parallel = False
check_distribution = False
demean_factors = False

if 'kz272' in os.path.expanduser('~'):
    code_path = '/gpfs/loomis/home.grace/kz272/scratch60/MSRR/Code/Output'
    data_path = '/gpfs/loomis/home.grace/kz272/scratch60/MSRR/Code/Empirical/Data'
elif 'kyzhou' in os.path.expanduser('~'):
    code_path = '/Users/kyzhou/Dropbox/MSRR/Code/Output'
    data_path = '/Users/kyzhou/Dropbox/MSRR/Code/Empirical/Data'
elif 'malamud' in os.path.expanduser('~'):
    code_path = '/Users/malamud/Dropbox/MY_STUFF/RESEARCH/MSRR/Code/SemyonOutput'
    data_path = '/Users/malamud/Dropbox/MY_STUFF/RESEARCH/MSRR/Code/Empirical/Data'
else:
    code_path = 'SemyonOutput'

if not os.path.exists(code_path):
    os.mkdir(code_path)


def random_fourier_features(signals, number_features):
    """
    signals are assumed to be T \times M
    :param signals:
    :param number_features:
    :return:
    """
    number_signals = signals.shape[1]
    np.random.seed(10)  # just fix a seed for replicability
    random_vectors = np.random.randn(number_signals, number_features) # number_signals \times number_features
    multiplied_features = np.matmul(signals.values, random_vectors)
    fourier_features_cos = np.cos(multiplied_features) # cos(gamma*wtmp*X+btmp*ones(1,T));
    fourier_features_sin = np.sin(multiplied_features)
    fourier_features = np.zeros([signals.shape[0], 2 * number_features])
    fourier_features[:, ::2] = fourier_features_cos
    fourier_features[:, 1::2] = fourier_features_sin
    return fourier_features

def random_fourier_features_uniformb(signals, number_features, gamma_):
    """
    signals are assumed to be T \times M
    :param signals:
    :param number_features:
    :return:
    """
    number_signals = signals.shape[1]
    T_ = signals.shape[0]
    np.random.seed(10)  # just fix a seed for replicability
    Wtmp = np.random.randn(number_features, number_signals)
    np.random.seed(10)  # just fix a seed for replicability
    btmp = np.random.randn(number_features, 1)

    multiplied_features = (gamma_ * np.matmul(Wtmp, signals.values.T) + np.matmul(btmp, np.ones((1,T_)))).T
    fourier_features_cos = np.cos(multiplied_features) # cos(gamma*wtmp*X+btmp*ones(1,T));
    return fourier_features_cos


def regression_with_tstats(predicted_variable, explanatory_variables):
    '''
    Get t-stats from regression
    :param predicted_variable:
    :param explanatory_variables:
    :return:
    '''
    x_ = explanatory_variables
    x_ = sm.add_constant(x_)
    y_ = predicted_variable
    # Newey-West standard errors with maxlags
    z_ = x_.copy().astype(float)

    result = sm.OLS(y_, z_).fit(cov_type='HAC', cov_kwds={'maxlags': 10})
    try:
        tstats = np.round(result.summary2().tables[1]['z'], 4)  # alpha t-stat (because for 'const')
        coef = np.round(result.summary2().tables[1]['Coef.'], 4)  # alpha value
        # tstat.index = list(z_.columns) z_ is an array

    except:
        print(f'something is wrong for t-stats')
    return tstats, coef


def GW_factor_construction(df):
    '''
    This function constructs factors in Goyal and Welch (2008)
    :param in_f:
    :return:
    '''

    # returns
    df['CRSP_SPvw_minus_Rfree'] = df['CRSP_SPvw'] - df['Rfree']

    # define dp/dy/ep/de
    df['dp'] = np.log(df['D12']) - np.log(df['Index'])
    df['dy'] = np.log(df['D12']) - np.log(df['Index'].shift(1))
    df['ep'] = np.log(df['E12']) - np.log(df['Index'])
    df['de'] = np.log(df['D12']) - np.log(df['E12'])

    # define default yield spread
    df['dfy'] = df['BAA'] - df['AAA']
    df['dfr'] = df['corpr'] - df['ltr']

    # Term Spread
    df['tms'] = df['lty'] - df['tbl']

    # order of AW table 1 + 'csp'
    # The available monthly predictors are:
    # dfy = BAA - AAA, infl, svar, de = log(D12) - log(E12), lty, tms = lty - tbl, tbl, dfr = corpr - ltr, dp = log(D12) - log(index), dy =
    # log(D12) - log(lagged index), ltr, ep = log(E12) - log(index), bm, ntis, csp
    columns_GW_Table1 = ['dfy', 'infl', 'svar', 'de', 'lty', 'tms', 'tbl', 'dfr', 'dp', 'dy', 'ltr', 'ep', 'b/m', 'ntis', 'csp']
    df = df[columns_GW_Table1 + ['CRSP_SPvw_minus_Rfree']]

    # shift factors by lag 1
    df[columns_GW_Table1] = df[columns_GW_Table1].shift(1)
    columns_GW_Table1_lag1 = [c + '_lag1' for c in columns_GW_Table1]
    df.columns = columns_GW_Table1_lag1 + ['CRSP_SPvw_minus_Rfree']

    df = df.loc[:, np.isnan(df).mean() < .5].dropna(axis=0)  # at least 50% of dates have data for the signals
    df = df.dropna()

    # save the data
    file_name = data_path + '/Amit Goyal/GW_factors_lag1.csv'
    if not os.path.exists(file_name):
        df.to_csv(file_name)

    returns = df.pop('CRSP_SPvw_minus_Rfree')
    Month = df.index
    Factors = df

    # if demean_factors:
    #     Factors = (Factors - Factors.rolling(100).mean().shift(1)).fillna(0)
    #     # CAREFUL: MAKE SURE WE ARE NOT USING FIRST 100 PERIODS !!!
    #
    # # normalize returns. This ensures that Sigma_eps =1
    # if normalize_returns:
    #     returns = returns / returns.rolling(12).std().shift(1)

    return returns, Factors, Month

def load_amit_goyal_data(demean_factors=False, normalize_returns = True):
    '''
    This function load and process the data from Goyal and Welch (2008)
    :return:
    '''
    # load data
    df = pd.read_excel(data_path + '/Amit Goyal/PredictorData2020.xlsx', index_col=0).sort_index()

    # construct the GW factors
    returns, Factors, Month = GW_factor_construction(df)

    if demean_factors:
        Factors = (Factors - Factors.rolling(100).mean().shift(1)).fillna(0)
        # CAREFUL: MAKE SURE WE ARE NOT USING FIRST 100 PERIODS !!!

    # normalize returns. This ensures that Sigma_eps =1
    if normalize_returns:
        returns = returns / returns.rolling(12).std().shift(1)
    return returns, Factors, Month


def xi_as_a_function_of_m(m_, c_, list_of_z):
    """
    THIS FUNCTION IS FOR N= 1!!!!
    :param m_:
    :param c_:
    :return:
    """

    z_vector = np.array(list_of_z).reshape(-1, 1)
    m_ = m_.reshape(-1, 1)
    xi = (1 - z_vector * m_) / ((1 / c_) - 1 + z_vector * m_)
    return xi


def xi_prime_as_a_function_of_m(m_prime, m_, xi, c_, list_of_z):
    """
    THIS FUNCTION IS FOR N= 1!!!!
    :param m_: 
    :return: 
    """
    z_vector = np.array(list_of_z).reshape(-1, 1)
    m_ = m_.reshape(-1, 1)
    xi_prime = c_ * (z_vector * m_prime.reshape(-1, 1) - m_) * ((1 + xi.reshape(-1, 1)) ** 2)
    return xi_prime


def all_betas_of_z_from_ridge(signals, returns, list_of_z, plot=False, eigenvalue_permutation=None):
    '''
    Thhis function computes betas for all z
    :param signals:
    :param returns:
    :param list_of_z:
    :return:
    '''

    signals = signals.T  # Change the dim of signals to M \times T
    T_ = signals.shape[1]

    c_M = signals.shape[0] / signals.shape[1]

    a_matrix = np.matmul(signals, signals.T) / T_  # M \times M
    eigval, eigvec = np.linalg.eigh(a_matrix)  # a_matrix = eigvec * diag(eigval) * eigvec.T

    # test effective ranks in Bartlett, Peter L., Philip M. Long, Gábor Lugosi, and Alexander Tsigler.
    # "Benign overfitting in linear regression." Proceedings of the National Academy of Sciences 117, no. 48 (2020): 30063-30070.
    # select_eigval_by_k_star = False
    # if select_eigval_by_k_star:
    #     r_k_Sigma = np.cumsum(eigval) / eigval
    #     EIGENVALUE_THRESHOLD_by_k_star = np.max(eigval[r_k_Sigma > signals.shape[0] * signals.shape[1]])
    #     EIGENVALUE_THRESHOLD = EIGENVALUE_THRESHOLD_by_k_star

    # only eigenvalues > 10e-10
    excess_number = max(signals.shape[0] - T_, 0)  # these guys will be pure zeros
    zero_eigenvalues = eigval[eigval < EIGENVALUE_THRESHOLD]
    top_eigenvectors = None

    if (len(zero_eigenvalues) > excess_number) or eigenvalue_permutation:
        # in this case, the lowest len(zero_eigenvalues) - excess_number eigenvalues
        # are there not because T_ < signals.shape[0], but because signals are degenerate and highly correlated
        top_eigenvectors = eigvec[:, (len(zero_eigenvalues) - excess_number):]
        if eigenvalue_permutation:
            # the next line is just to make sure that the length of the permutation agrees
            eigenvalue_permutation = eigenvalue_permutation[(len(zero_eigenvalues) - excess_number):]
            top_eigenvectors = top_eigenvectors[:, eigenvalue_permutation]
        rotated_signals = np.matmul(top_eigenvectors.T, signals)
        signals = rotated_signals
        c_M = signals.shape[0] / signals.shape[1]
        a_matrix = np.matmul(signals, signals.T) / T_  # M \times M

        eigval, eigvec = np.linalg.eigh(a_matrix)

    num_signals = len(eigval)
    # beta = (zI + A)^{-1} (S*R) = V * (zI+lambda)^{-1} V' * (SR)

    signal_times_return = np.matmul(signals, returns) / T_  # This is (SR): M \times 1
    signal_times_return_times_v = np.matmul(eigvec.T, signal_times_return)  # this is V' * (SR) # min(T,M) \times 1

    breakpoint()

    # m function
    m_ = np.array([np.sum(1 / (eigval + z)) / num_signals for z in list_of_z])
    m_prime = np.array([np.sum(1 / (eigval + z) ** 2) / num_signals for z in list_of_z])

    xi = xi_as_a_function_of_m(m_, c_M, list_of_z)
    xi_prime = xi_prime_as_a_function_of_m(m_prime, m_, xi, c_M, list_of_z)

    all_theoretical_functions = {'m': m_, 'm_prime': m_prime, 'xi': xi, 'xi_prime': xi_prime,
                                 'top_eigenvectors': top_eigenvectors}

    # this is (zI+lambda)^{-1} V' * (SR)
    betas = [np.matmul(np.diag(1 / (eigval + z), 0), signal_times_return_times_v) for z in list_of_z]
    betas = np.concatenate(betas, axis=1)  # M \times len(z)

    # finally beta = V * (zI+lambda)^{-1} V' * (SR)
    betas = np.matmul(eigvec, betas)  # This is time consuming, so do it once. M \times len(z)

    predicted_returns_for_all_values_of_z = np.matmul(signals.T, betas)  # T_OOS \times len(z)

    # this is prediction error, R_{t+1} - E_t[R]
    prediction_errors_for_all_z_values = returns.reshape(-1, 1) - predicted_returns_for_all_values_of_z
    null = (returns ** 2).mean()
    in_sample_mse_across_z = (prediction_errors_for_all_z_values ** 2).mean(0)
    in_sample_r_squared_across_z = 1 - in_sample_mse_across_z / null
    if in_sample_r_squared_across_z[0] < 0:
        print(f'hmm, even in sample we do not have much predictability with an r squared of '
              f'{in_sample_r_squared_across_z[1]}')
    if plot:
        plot_logm_logz(list_of_z, m_, c_M)
        plot_zm_z(list_of_z, m_, c_M)
    return betas, all_theoretical_functions, signals.T, c_M


def theoretical_mse(all_theoretical_functions, list_of_z, c_M, b_star):
    '''
    complete formula (42), with arbitrary b_* , and of course Sigma=1
    The first two terms of Eq.(42) is N according to Eq. (43)
    :param all_theoretical_functions:
    :param list_of_z:
    :param c_M:
    :param b_star:
    :return:
    '''

    N = 1
    xi = all_theoretical_functions['xi']  # len(z) \time 1
    xi_prime = all_theoretical_functions['xi_prime']  # len(z) \time 1
    last_term = xi_prime * (list_of_z.reshape(-1, 1) - list_of_z.reshape(-1, 1) ** 2 / c_M * N * b_star)
    return N + xi + last_term


def get_b_star(list_of_z, c_M, tr_Sigma_hat, empirical_MSE, all_theoretical_functions):
    z_array = list_of_z.reshape(-1, 1)
    denominator = z_array ** 2 / c_M * tr_Sigma_hat
    numerator = z_array - (empirical_MSE - N - all_theoretical_functions['xi']) / all_theoretical_functions['xi_prime']

    b_star_array = numerator / denominator
    b_star = np.median(b_star_array)  # np.mean(b_star_array)
    return b_star, b_star_array


def oos_performance(oos_signals, oos_returns, betas_across_z_values, OOSMeanReturn):
    """
    oos_signals = T \times M; betas = M \times len(z_values)
    :param oos_signals: dim = (T, M)
    :param oos_returns: dim = (T, rho) or (T, 1)
    :param betas_across_z_values: dim = (M, len(z))
    :param OOSMeanReturn: a constant
    :return:
    """
    # the next object is T \times len(z_values)

    # this is the expected return, E_t[R]
    predicted_returns_for_all_values_of_z = np.matmul(oos_signals, betas_across_z_values)  # T_OOS \times len(z)

    # this is prediction error, R_{t+1} - E_t[R]
    prediction_errors_for_all_z_values = oos_returns.reshape(-1, 1) - predicted_returns_for_all_values_of_z
    null = (oos_returns ** 2).mean()
    mse_across_z = (prediction_errors_for_all_z_values ** 2).mean(0)
    r_squared_across_z = 1 - mse_across_z / null

    # the next line is predRet_t * R_{t+1} = (S_t*betaEstimate) * R_{t+1}
    managed_returns_across_z_values = predicted_returns_for_all_values_of_z * oos_returns.reshape(-1, 1)

    # ManagedReturns = TimingFactor_t  OOSMeanReturn +  TimingFactor_t  DemeanedReturns
    # TimingFactor_t OOSMeanReturn:
    TimingFactor_times_OOSMeanReturn_across_z_values = predicted_returns_for_all_values_of_z * OOSMeanReturn

    # now we compute alpha
    alpha_tstats = np.array([regression_with_tstats(managed_returns_across_z_values[:, i], oos_returns.reshape(-1, 1))[0]['const']
                 for i in range(managed_returns_across_z_values.shape[1])])
    # we probably only need the alpha coef; but maybe beta as some point is worth investigating

    alpha_coefs = np.array(
        [regression_with_tstats(managed_returns_across_z_values[:, i], oos_returns.reshape(-1, 1))[1][1]
         for i in range(managed_returns_across_z_values.shape[1])])

    ptf_minus_coef_times_mkt = managed_returns_across_z_values - np.matmul(oos_returns.reshape(-1, 1), alpha_coefs.reshape(-1, 1).T)
    ptf_minus_coef_times_mkt_returns = ptf_minus_coef_times_mkt.mean(axis=0)
    ptf_minus_coef_times_mkt_risks = (ptf_minus_coef_times_mkt ** 2).mean(axis=0)
    ptf_minus_coef_times_mkt_sharpes = ptf_minus_coef_times_mkt_returns / np.sqrt(ptf_minus_coef_times_mkt_risks)

    oos_returns_across_z_values = managed_returns_across_z_values.mean(axis=0)
    oos_risks_across_z_values = (managed_returns_across_z_values ** 2).mean(axis=0)
    oos_sharpes_across_z_values = oos_returns_across_z_values / np.sqrt(oos_risks_across_z_values)

    TimingFactor_times_OOSMeanReturn_returns_across_z_values = TimingFactor_times_OOSMeanReturn_across_z_values.mean(axis=0)
    TimingFactor_times_OOSMeanReturn_risks_across_z_values = (TimingFactor_times_OOSMeanReturn_across_z_values**2).mean(
        axis=0)
    TimingFactor_times_OOSMeanReturn_sharpes_across_z_values = TimingFactor_times_OOSMeanReturn_returns_across_z_values/np.sqrt(TimingFactor_times_OOSMeanReturn_risks_across_z_values)

    performance = {'r_squared_across_z': r_squared_across_z,
                   'oos_sharpes_across_z_values': oos_sharpes_across_z_values,
                   'oos_returns_across_z_values': oos_returns_across_z_values,
                   'oos_risks_across_z_values': oos_risks_across_z_values,
                   'mse_across_z': mse_across_z, # 'alpha_regression_on_market': alpha_coefs,
                   't-stat_regression_on_market': alpha_tstats,
                   'OOSMeanReturn_ptf_sharpes': TimingFactor_times_OOSMeanReturn_sharpes_across_z_values,
                   'OOSMeanReturn_ptf_returns': TimingFactor_times_OOSMeanReturn_returns_across_z_values,
                   'OOSMeanReturn_ptf_risks': TimingFactor_times_OOSMeanReturn_risks_across_z_values,
                   'ptf_minus_coef_times_mkt_sharpes': ptf_minus_coef_times_mkt_sharpes,
                   'managed_returns_across_z_values': managed_returns_across_z_values
                   }
    return performance


def produce_POLY_Factors(Factors, subset_of_signals):
    '''
    This function produces input data pieces for function run_analysis_of_a_signal_subset
    :param Factors_input_training:
    :param Factors_input_OOS:
    :param subset_of_signals:
    :return:
    '''
    if isinstance(subset_of_signals, tuple):
        subset_of_signals = list(subset_of_signals)
    selected_factors = Factors.values[:, subset_of_signals]
    poly = PolynomialFeatures(degree=DEGREE, interaction_only=True)
    selected_factors = poly.fit_transform(selected_factors)[:, 1:]  # [:, 1:] is to get rid of the vector of ones
    return selected_factors


def produce_data_subsample(Factors, subset_of_signals, POLY=True):
    '''
    This function produces input data pieces for function run_analysis_of_a_signal_subset
    :param Factors_input_training:
    :param Factors_input_OOS:
    :param subset_of_signals:
    :return:
    '''
    if isinstance(subset_of_signals, tuple):
        subset_of_signals = list(subset_of_signals)
    if isinstance(Factors, pd.DataFrame):
        Factors = Factors.values
    selected_factors = Factors[:, subset_of_signals]
    if POLY:
        poly = PolynomialFeatures(degree=DEGREE, interaction_only=True)
        selected_factors = poly.fit_transform(selected_factors)[:, 1:]  # [:, 1:] is to get rid of the vector of ones
    return selected_factors[:T_TRAIN, :], selected_factors[T_TRAIN:, :]


def demean_and_standardize_using_in_sample(in_sample_signals, oos_signals):
    in_sample_means = in_sample_signals.mean(0)
    in_sample_stds = in_sample_signals.std(0)

    in_sample_signals -= in_sample_means.reshape(1, -1)
    in_sample_signals /= in_sample_stds.reshape(1, -1)

    oos_signals -= in_sample_means.reshape(1, -1)
    oos_signals /= in_sample_stds.reshape(1, -1)
    return in_sample_signals, oos_signals


def run_analysis_of_a_signal_subset(signal_subset, returns_training, returns_oos, OOSMeanReturn, plot=False,
                                    demean_and_standardize_signals=DEMEAN_AND_STANDARDIZE_SIGNALS):
    '''
    This function get the performance of a signal subset
    :param signal_subset:
    :param returns_training:
    :param returns_oos: T_OOS \times 1
    :return: (mse_across_z, r_squared_across_z, oos_sharpes_across_z_values,
            oos_returns_across_z_values, oos_risks_across_z_values)
    '''
    in_sample_signals, oos_signals = signal_subset
    if demean_and_standardize_signals:
        in_sample_signals, oos_signals = demean_and_standardize_using_in_sample(in_sample_signals, oos_signals)

    all_betas, all_theoretical_functions, in_sample_signals, c_M = \
        all_betas_of_z_from_ridge(in_sample_signals, returns_training,
                                  list_of_z, plot=plot)
    beta_2nd_norm = LA.norm(all_betas, axis=0)

    if all_theoretical_functions['top_eigenvectors'] is not None:
        # this means that there we degeneracies in the in-sample signals and hence we had to take care of it
        # we have reduced the dimension, and now we need to do the same with the oos_eigenvectors
        top_eigenvectors = all_theoretical_functions['top_eigenvectors']
        oos_signals = np.matmul(top_eigenvectors.T, oos_signals.T).T
    performance = oos_performance(oos_signals, returns_oos, all_betas, OOSMeanReturn)
    performance['beta_2nd_norm'] = beta_2nd_norm

    # Simulated performance
    # T_OOS = oos_signals.shape[0]
    # simulate_oos_signals, simulate_oos_returns, z_star_idx = Simulate_OOS_Signals_and_Returns(in_sample_signals,
    #                                                                                           all_betas, T_OOS, rho)
    # simulated_performance = {}
    # for i in range(len(rho)):
    #     key = 'rho%s' % (rho[i])
    #     simulated_performance[key] = oos_performance(simulate_oos_signals, simulate_oos_returns[:, i].reshape(-1, 1),
    #                                                  all_betas, OOSMeanReturn)
    # performance['simulated_performance'] = simulated_performance

    # record c_M
    # c_M = oos_signals.shape[1] / in_sample_signals.shape[0]  # careful here, we are using oos_signals.shape[1]

    # because this might have changed due to the degeneracy issues
    performance['effective_c_M'] = c_M

    # print(f'hey, we just ran with an effective c_M {c_M} '
    #       f'and the ridgeless Rsquared is {performance["r_squared_across_z"][0]}; \n alphas '
    #       f'are given by {performance["t-stat_regression_on_market"]}')

    # # get b_star for each z by mse_empirical
    # mse_empirical = performance['mse_across_z'].reshape(-1, 1)
    # b_star_mean, b_star_array = get_b_star(list_of_z, c_M, N, mse_empirical, all_theoretical_functions)
    # if not os.path.exists(code_path + '/list_of_b_star/log10(b_star) vs. log10(z), c_M = %.4f' %(c_M) + '.png'):
    #     plot_b_star(list_of_z.reshape(-1,1), b_star_array.flatten(), c_M)
    #
    # for b_star in [0.01, 1, 10, 100, 1000]:
    #     mse_theoretical = theoretical_mse(all_theoretical_functions, list_of_z, c_M, b_star)
    #     plot_TheoreticalMSE_logz(list_of_z, mse_theoretical, c_M, b_star)
    #     best_z = list_of_z[np.argmin(mse_theoretical)]
    #     best_beta = all_betas[:, np.argmin(mse_theoretical)]
    #     best_b_star = (all_betas[:, np.argmin(mse_theoretical)] ** 2).sum()
    #     print(f'{b_star, best_z, best_b_star}')
    #     print('mean(mse_theoretical) = %s' % (mse_theoretical))
    return performance

def produce_data_slices(returns, Factors, STARTING_POINT):
    """
    just cut data slices over a rolling window
    :param returns:
    :param Factors:
    :param T_train:
    :param T_OOS:
    :return:
    """
    data_pieces = [(returns.iloc[starting_:(starting_ + T_TRAIN + T_OOS)],
                    Factors.iloc[starting_:(starting_ + T_TRAIN + T_OOS), :])
                   for starting_ in range(STARTING_POINT, returns.shape[0] - (T_TRAIN + T_OOS) + 1, 12)]
    return data_pieces


def results_over_expanding_signals_set(expanded_signals, returns_training, returns_oos, OOSMeanReturn):
    results_collect = []
    plot = False  # True  # if True, we plot log scale m(z), but it's very time-consuming

    for number_of_base_signals in range(0, expanded_signals.shape[1], M_step):
        print('number of signals = %s' % (number_of_base_signals + 1))

        samples = [tuple(range(number_of_base_signals + 1))]
        all_signal_subsamples = [(produce_data_subsample(expanded_signals, subset_of_signals, False),
                                  returns_training, returns_oos, OOSMeanReturn, plot)
                                 for subset_of_signals in samples]
        # each element has [(Factors_input_training, Factors_input_OOS), returns_training, returns_oos]
        # results_collect['M'].append(all_signal_subsamples[0][0][0].shape[1])
        parallel = False
        if parallel:
            pool = mp.Pool(mp.cpu_count() - 2)
            results = pool.starmap(run_analysis_of_a_signal_subset, all_signal_subsamples)
            pool.close()
            pool.join()
        else:
            # results = np.array([run_analysis_of_a_signal_subset(*data_piece) for data_piece in all_signal_subsamples])
            results = [run_analysis_of_a_signal_subset(*data_piece) for data_piece in all_signal_subsamples]

        # results_collect.append(results)
        results_collect = results_collect + results
    return results_collect


def run_analysis_of_a_time_slice_select_signals_continuously(data_slice):
    """
    Function run_analysis_of_a_time_slice with continuous c
    :param data_slice:
    :return: each element in results_collect is a dictionary with dict_keys(['r_squared_across_z', 'oos_sharpes_across_z_values',
    'oos_returns_across_z_values', 'oos_risks_across_z_values', 'mse_across_z', 'simulated_performance', 'effective_c_M'])

    results_collect[i]['simulated_performance'].keys() = dict_keys(['rho0.01', 'rho0.02', 'rho0.03', 'rho0.04'])

    results_collect[i]['simulated_performance']['rho0.02'].keys() = dict_keys(['r_squared_across_z',
    'oos_sharpes_across_z_values', 'oos_returns_across_z_values', 'oos_risks_across_z_values', 'mse_across_z'])
    """
    returns = data_slice[0]
    Factors = data_slice[1]

    train_index = range(0, T_TRAIN)
    OOS_index = range(T_TRAIN, T_TRAIN + T_OOS)

    # Get training and OOS period
    train_start_month = returns.index[0]
    print('Starting data is %s' % (train_start_month))
    train_end_month = returns.index[T_TRAIN - 1]
    OOS_start_month = returns.index[T_TRAIN]
    OOS_end_month = returns.index[T_TRAIN + T_OOS - 1]
    train_and_OOS_str = 'Train %s - %s, OOS %s - %s' % (
    train_start_month, train_end_month, OOS_start_month, OOS_end_month)

    # returns for training and OOS
    returns_training = returns.iloc[list(train_index)].values.reshape(-1, 1)
    returns_oos = returns.iloc[list(OOS_index)].values.reshape(-1, 1)
    OOSMeanReturn = np.mean(returns_oos)

    if DEMEAN_RETURNS_SEPARATELY_INSAMPLE_OOS:
        returns_training = returns_training - np.mean(returns_training)
        returns_oos = returns_oos - OOSMeanReturn

    # SRU of the market
    SRU_SP500 = np.mean(returns_oos) / np.sqrt(np.mean(returns_oos ** 2))

    permute_eigenvectors = PERMUTE_EIGENVECTORS

    # for loop for each M
    if not USE_FOURIER_FEATURES:
        expanded_signals = produce_POLY_Factors(Factors, range(0, Factors.shape[1]))  # dim = (400, 1092)
    else:
        expanded_signals = random_fourier_features(Factors, number_features=number_of_PC_signals)
        permute_eigenvectors = False

    # now rotate by PC if necessary
    if permute_eigenvectors:
        print('permute eigenvectors')
        expanded_signals_cov = np.matmul(expanded_signals.T, expanded_signals)
        eigval, eigvec = np.linalg.eigh(expanded_signals_cov)
        expanded_signals = np.matmul(eigvec[:, range(eigvec.shape[1]- number_of_PC_signals, eigvec.shape[1])].T, expanded_signals.T).T

    # generate 1000 random permutations
    if number_of_combinations == 1 or USE_FOURIER_FEATURES:
        all_permutations = [range(expanded_signals.shape[1])]
    else:
        all_permutations = [np.random.RandomState(seed=i).permutation(expanded_signals.shape[1]) for i in range(number_of_combinations)]

    randomly_permuted_signals_sets = [(expanded_signals[:, perm], returns_training, returns_oos, OOSMeanReturn) for perm in all_permutations]
    # now we just loop through all these randomly permuted guys
    # all_results = [(results_over_expanding_signals_set(randomly_permuted_signals, returns_training, returns_oos))
    #                for randomly_permuted_signals in randomly_permuted_signals_sets]
    parallel = False
    if parallel:
        # Grace: os.environ[‘SLURM_CPUS_PER_TASK']
        if 'kz272' in os.path.expanduser('~'):
            # Grace
            pool = mp.Pool(int(os.environ['SLURM_CPUS_PER_TASK']))
        else:
            # Local
            pool = mp.Pool(mp.cpu_count() - 1)
        print(randomly_permuted_signals_sets)
        all_results = pool.starmap(results_over_expanding_signals_set, randomly_permuted_signals_sets)
        pool.close()
        pool.join()
    else:
        all_results = [results_over_expanding_signals_set(*randomly_permuted_signals)
                       for randomly_permuted_signals in randomly_permuted_signals_sets]

    results_dict = {'results': all_results, 'SRU_SP500': SRU_SP500, 'train_and_OOS_str': train_and_OOS_str, 'OOSMeanReturn': OOSMeanReturn}
    return results_dict

def parameter_strings_for_output(starting_point_idx):
    '''
    Produce strings for data saving and parameters
    :param starting_point_idx:
    :return:
    '''
    para_str = 'start_idx %s, T %s, T_OOS %s, num_combinations %s, EIGEN_THRESH %s, DEGREE %s, PC_signals %s, normalize_returns %s, FOURIER %s' \
               % (starting_point_idx, T_TRAIN, T_OOS, number_of_combinations, DEGREE, PERMUTE_EIGENVECTORS, number_of_PC_signals, normalize_returns, USE_FOURIER_FEATURES)
    para_str_title = 'T %s, number of combinations %s, T_OOS %s, DEGREE %s \n PERMUTE_EIGENVECTORS %s, PC_signals %s, \n NORMALIZE_RETURNS %s, FOURIER_FEATURES %s' \
                     % (T_TRAIN, number_of_combinations, T_OOS, DEGREE, PERMUTE_EIGENVECTORS, number_of_PC_signals, normalize_returns, USE_FOURIER_FEATURES)
    return para_str, para_str_title


def final_plotting_block(c_M, results, SRU_500_array, performance_mean_among_samples_and_data_slices, para_str_title, para_str):
    save_path = os.path.join(code_path, 'Empirical_avg_across_data_slices')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = os.path.join(save_path, 'Plots_PC_signals%s_T%s_T_OOS%s_DEGREE%s_Fourier%s_NormRet%s' % (number_of_PC_signals, T_TRAIN, T_OOS, DEGREE, USE_FOURIER_FEATURES, normalize_returns))
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path = os.path.join(save_path, para_str)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    z_use = range(0, len(list_of_z))
    for i, title_str in enumerate(performance_names):
        series_refined_plot(save_path, c_M, performance_mean_among_samples_and_data_slices[:, i, z_use], title_str,
                    list_of_z[z_use], para_str_title, para_str, SRU_500_array)
        series_plot(save_path, c_M, performance_mean_among_samples_and_data_slices[:, i, z_use], title_str,
                            list_of_z[z_use], para_str_title, para_str, SRU_500_array)
        # std_plot(save_path, c_M, performance_std_among_samples_and_data_slices[:, i, z_use], title_str, list_of_z[z_use],
        #             para_str_title, para_str)

    ##################### check distributions #####################
    if check_distribution:
        # Plot
        save_path = os.path.join(save_path, 'Distributions')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for i, title_str in enumerate(performance_names):
            print(title_str)
            for j, z in enumerate(list_of_z):
                Dist = np.array([perform_i[:, i, j] for perform_i in results])
                boxplot(save_path, c_M, T_TRAIN, Dist, title_str + ' Distribution', para_str_title, para_str, True, z)
                # boxplot(save_path, c_M, T, Dist, title_str + ' Distribution', para_str_title, para_str, False, z)

        ##################### Scatter plot of returns and risks #####################
        for i, c in enumerate(c_M):
            for j, z in enumerate(list_of_z):
                scatterplot(save_path, c, results[i][:, 2, j], results[i][:, 3, j], para_str_title, para_str, z)


def produce_performance_summaries(results, para_str_title, para_str):

    SRU_500_array = np.array([result_i['SRU_SP500'] for result_i in results])
    train_and_OOS_str_array = np.array([result_i['train_and_OOS_str'] for result_i in results])
    OOSMeanReturn_array = np.array([result_i['OOSMeanReturn'] for result_i in results])

    results = np.array([result_i['results'] for result_i in results])

    results_array = results.flatten()  # dim = (len(data_slices) * len(signals_combinations) * len(M_list), )
    results_df = pd.DataFrame(list(results_array))  # shape = (len(data_slices) * len(signals_combinations) * len(M_list), 7)
    all_unique_c_m = np.sort(results_df['effective_c_M'].unique())

    ##################### Plot for R-squared, SRU, Returns, Risks along c = M/T #####################
    para_str_title = para_str_title + '\n' + train_and_OOS_str_array[0]
    para_str = para_str + train_and_OOS_str_array[0]

    # get the performance mean among samples and data slices

    performance_mean = []
    performance_std = []
    performance_columns = ['r_squared_across_z', 'oos_sharpes_across_z_values',
       'oos_returns_across_z_values', 'oos_risks_across_z_values',
       'mse_across_z', 't-stat_regression_on_market',
       'OOSMeanReturn_ptf_sharpes', 'OOSMeanReturn_ptf_returns',
       'OOSMeanReturn_ptf_risks', 'ptf_minus_coef_times_mkt_sharpes',
       'beta_2nd_norm']

    for c_M_i in all_unique_c_m:
        df_i = results_df[results_df.effective_c_M == c_M_i]
        oos_empirical_performance_i = np.array(df_i[performance_columns].values.tolist())
        oos_empirical_performance_i_mean = np.vstack(np.mean(oos_empirical_performance_i, axis=0))
        oos_empirical_performance_i_std = np.vstack(np.std(oos_empirical_performance_i, axis=0))
        performance_mean.append(oos_empirical_performance_i_mean)

    performance_mean_among_samples_and_data_slices = np.array(
        performance_mean)  # dim = (len(c_M), len(performance_names), len(z))
    performance_std_among_samples_and_data_slices = np.array(performance_std)

    return all_unique_c_m, SRU_500_array, performance_mean_among_samples_and_data_slices, \
           performance_std_among_samples_and_data_slices, para_str_title, para_str


def run_a_loop_through_data_slices(data_slices):
    # If the result file doesn't exist, then compute the result
    parallel = False
    if DISCRETE_SIGNALS_EXPANSION:
        # DISCRETE_SIGNALS_EXPANSION = True
        if parallel:
            pool = mp.Pool(mp.cpu_count() - 2)
            results = pool.map(run_analysis_of_a_time_slice, data_slices)
            pool.close()
            pool.join()
        else:
            results = [run_analysis_of_a_time_slice(data_piece) for data_piece in data_slices]
    else:
        if parallel:
            pool = mp.Pool(mp.cpu_count() - 2)
            results = pool.map(run_analysis_of_a_time_slice_select_signals_continuously, data_slices)
            pool.close()
            pool.join()
        else:
            results = [run_analysis_of_a_time_slice_select_signals_continuously(data_piece) for data_piece in
                       data_slices]

    results = np.array(results)  # len(data_slices) \times len(signals)
    print("--- %s seconds ---" % (time.time() - start_time))
    return results

#####################################################################################################################
####################### The following functions are only used for discrete signals expansions #######################
#####################################################################################################################

def build_samples(size, number_of_combinations, number_of_all_Factors):
    '''
    Get combinations of size out of range(0, number_of_all_Factors)
    :param size:
    :param number_of_combinations:
    :param number_of_all_Factors:
    :return:
    '''

    # get the combinations
    comb_list = list(combinations(range(0, number_of_all_Factors), size))

    # signals_adding_methods = ['poly order', 'eigval descending', 'eigval ascending', 'random']
    # Get signals combination index
    if len(comb_list) <= number_of_combinations:
        samples = comb_list
    else:
        # if we cross number_of_combinations then we just randomly sample
        samples = random.choices(comb_list, k=number_of_combinations)
    return samples


# def build_random_permutations(number=1000, signals):
#     '''
#     Get combinations of size out of range(0, number_of_all_Factors)
#     :param size:
#     :param number_of_combinations:
#     :param number_of_all_Factors:
#     :return:
#     '''
#
#     # get the combinations
#     permutations = np.random.permutation()
#
#     return samples


def run_analysis_of_a_time_slice(data_slice):
    """
    analyze a given time slice with discrete c
    :param data_slice:
    :return: each element in results_collect is a dictionary with dict_keys(['r_squared_across_z', 'oos_sharpes_across_z_values',
    'oos_returns_across_z_values', 'oos_risks_across_z_values', 'mse_across_z', 'simulated_performance', 'effective_c_M'])

    results_collect[i]['simulated_performance'].keys() = dict_keys(['rho0.01', 'rho0.02', 'rho0.03', 'rho0.04'])

    results_collect[i]['simulated_performance']['rho0.02'].keys() = dict_keys(['r_squared_across_z',
    'oos_sharpes_across_z_values', 'oos_returns_across_z_values', 'oos_risks_across_z_values', 'mse_across_z'])
    """
    returns = data_slice[0]
    Factors = data_slice[1]

    train_index = range(0, T_TRAIN)
    OOS_index = range(T_TRAIN, T_TRAIN + T_OOS)
    # Get training and OOS period
    train_start_month = returns.index[0]
    print('Starting data is %s' % (train_start_month))

    # returns for training and OOS
    returns_training = returns.iloc[list(train_index)].values.reshape(-1, 1)
    returns_oos = returns.iloc[list(OOS_index)].values.reshape(-1, 1)

    # SRU of the market
    SRU_SP500 = np.mean(returns_oos) / np.sqrt(np.mean(returns_oos ** 2))

    results_collect = []
    plot = False  # True  # if True, we plot log scale m(z), but it's very time-consuming
    for number_of_base_signals in range(Factors.shape[1]):  # [12]:  # [8, 9, 10, 11, 12]:#
        print('number of signals = %s' % (number_of_base_signals + 1))

        # Get all possible subsets of lenth (i + 1)
        if number_of_combinations == 1:
            samples = [tuple(range(number_of_base_signals + 1))]  # fix the factors when number_of_
            # combinations == 1, for the check of newly added eigenvalues. So we are adding signals one-by-one
        else:
            samples = build_samples(number_of_base_signals + 1, number_of_combinations, Factors.shape[1])

        # produce data subsamples
        all_signal_subsamples = [(produce_data_subsample(Factors, subset_of_signals),
                                  returns_training, returns_oos, plot)
                                 for subset_of_signals in samples]
        # each element has [(Factors_input_training, Factors_input_OOS), returns_training, returns_oos]
        # results_collect['M'].append(all_signal_subsamples[0][0][0].shape[1])
        # parallel = False
        # if parallel:
        #     pool = mp.Pool(4)
        #     results = pool.starmap(run_analysis_of_a_signal_subset, all_signal_subsamples)
        #     pool.close()
        #     pool.join()
        # else:
        # results = np.array([run_analysis_of_a_signal_subset(*data_piece) for data_piece in all_signal_subsamples])
        # Note that parallel cannot be used here, because the order of c = /T matters
        results = [run_analysis_of_a_signal_subset(*data_piece) for data_piece in all_signal_subsamples]

        # results_collect.append(results)
        results_collect = results_collect + results

    results_dict = {'results': results_collect, 'SRU_SP500': SRU_SP500}

    return results_dict

#####################################################################################################################
#####################################################################################################################
#####################################################################################################################


if __name__ == '__main__':
    start_time = time.time()

    starting_point_idx = int(sys.argv[1])
    STARTING_POINT = 1128 - T_TRAIN - T_OOS - 12 * starting_point_idx

    # Load the data and process it
    returns, Factors, Month = load_amit_goyal_data(demean_factors, normalize_returns)

    data_slices = [produce_data_slices(returns, Factors, STARTING_POINT)[0]]

    ##################################################################
    para_str, para_str_title = parameter_strings_for_output(starting_point_idx)

    # Path for result save
    save_path_results = os.path.join(code_path, 'results_array')
    if not os.path.exists(save_path_results):
        os.mkdir(save_path_results)

    save_path_results1 = os.path.join(save_path_results, 'results_array_PC_signals%s_T%s_T_OOS%s_DEGREE%s_Fourier%s_NormRet%s' % (number_of_PC_signals, T_TRAIN, T_OOS, DEGREE, USE_FOURIER_FEATURES, normalize_returns))
    if not os.path.exists(save_path_results1):
        os.mkdir(save_path_results1)

    results_file = save_path_results1 + '/' + para_str + '_results.npy'  # 'aaaaaaa'
    if os.path.exists(results_file):
        # If the result file exists, then load the result
        with open(results_file, 'rb') as f:
            results = np.load(f, allow_pickle=True)
    else:
        results = run_a_loop_through_data_slices(data_slices)
        with open(results_file, 'wb') as f:
            np.save(f, results)

    all_unique_c_m, SRU_500_array, performance_mean_among_samples_and_data_slices, performance_std_among_samples_and_data_slices, para_str_title, para_str \
        = produce_performance_summaries(results, para_str_title, para_str)

    final_plotting_block(all_unique_c_m, results, SRU_500_array, performance_mean_among_samples_and_data_slices, para_str_title, para_str)