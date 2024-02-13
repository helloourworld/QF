# This file contains functions for running out-of-sample simulations with Sigma_epsilon.
import numpy as np
from auxilliary_functions import matrix_square_root
from Eigenvalue_functions import map_from_m_to_mPsi
from Psi_Sigma_Sigma_eps_functions import *
from expectation_functions import benchmark_SRU
import time
import matplotlib.pyplot as plt
import os
import pickle
from sklearn.metrics import r2_score


def generate_data(N, M, T, b, T_test, Sigma, Psi, Sigma_eps):
    """
    :param N: number of assets
    :param M: number of variables
    :param T: training samples
    :param b: the level of beta, b = ||beta||^2
    :param T_test: the number of test samples
    :param K_perc: percentile of non-zero elements in the diagnal of Sigma
    :param Sigma: covariance of signals across assets for S generation
    :param Psi: covariance across signals for S generation
    :param Sigma_eps:
    :return: beta, in_sample_signals, out_of_sample_signals, in_sample_returns, out_of_sample_returns
    """

    beta = np.random.normal(0, np.sqrt(b / M), size=(M, 1))

    # "raw" signals X_t, X has shape N*M*T
    # signals = np.random.randn(N, M, 2 * T)
    X = np.random.randn(T + T_test, N, M)

    # Signal S generation, S has dimension N*M
    Psi_sqrt = matrix_square_root(Psi)
    Sigma_sqrt = matrix_square_root(Sigma)

    # Expand Psi_sqrt and Sigma_sqrt in dimension T
    Psi_sqrt_T = np.repeat(Psi_sqrt[np.newaxis, :, :], T + T_test, axis=0)  # T*M*M
    Sigma_sqrt_T = np.repeat(Sigma_sqrt[np.newaxis, :, :], T+T_test, axis=0)  # T*N*N

    # Generate S
    S = np.matmul(np.matmul(Sigma_sqrt_T, X), Psi_sqrt_T).transpose((1, 2, 0))  # N*M*(T+T_test)

    # epsilon
    # epsilon = np.random.randn(N, T+T_test)
    epsilon = np.random.multivariate_normal(np.zeros(N), Sigma_eps, size=(T+T_test)).T

    # returns
    returns = np.tensordot(S, beta, axes=(1, 0)).squeeze() + epsilon  # (N, T + T_test)

    # in-sample
    in_sample_signals = S[:, :, :T]
    in_sample_returns = returns[:, :T]

    # OOS
    out_of_sample_signals = S[:, :, T:]
    out_of_sample_returns = returns[:, T:]
    return beta, in_sample_signals, out_of_sample_signals, in_sample_returns, out_of_sample_returns, epsilon


def generate_in_sample_data(N, M, T, b, Sigma, Psi, Sigma_eps):
    """
    :param N: number of assets
    :param M: number of variables
    :param T: training samples
    :param b: the level of beta, b = ||beta||^2
    :param K_perc: percentile of non-zero elements in the diagnal of Sigma
    :param Sigma: covariance of signals across assets for S generation
    :param Psi: covariance across signals for S generation
    :param Sigma_eps:
    :return: beta, in_sample_signals, out_of_sample_signals, in_sample_returns, out_of_sample_returns
    """

    beta = np.random.normal(0, np.sqrt(b / M), size=(M, 1))

    # "raw" signals X_t, X has shape N*M*T
    # signals = np.random.randn(N, M, 2 * T)
    X = np.random.randn(T, N, M)

    # Signal S generation, S has dimension N*M
    Psi_sqrt = matrix_square_root(Psi)
    Sigma_sqrt = matrix_square_root(Sigma)

    # Expand Psi_sqrt and Sigma_sqrt in dimension T
    Psi_sqrt_T = np.repeat(Psi_sqrt[np.newaxis, :, :], T, axis=0)  # T*M*M
    Sigma_sqrt_T = np.repeat(Sigma_sqrt[np.newaxis, :, :], T, axis=0)  # T*N*N

    # Generate S
    S = np.matmul(np.matmul(Sigma_sqrt_T, X), Psi_sqrt_T).transpose((1, 2, 0))  # N*M*(T+T_test)

    # epsilon
    # epsilon = np.random.randn(N, T+T_test)
    epsilon = np.random.multivariate_normal(np.zeros(N), Sigma_eps, size=(T)).T

    # returns
    returns = np.tensordot(S, beta, axes=(1, 0)).squeeze() + epsilon  # (N, T + T_test)

    # in-sample
    # in-sample
    in_sample_signals = S[:, :, :T]
    in_sample_returns = returns[:, :T]

    True_Rsq = r2_score(in_sample_returns.T, (returns - epsilon).T)

    return beta, in_sample_signals, in_sample_returns, True_Rsq


def build_factor_returns(out_of_sample_signals, out_of_sample_returns, Sigma_epsilon_hat):
    """
    This just does S_t'R_{t+1} but in a smart way
    :param out_of_sample_signals: out-of-sample signals with dimension N*M*T_test
    :param out_of_sample_returns: out-of-sample returns with dimension N*T_test
    :return: factor_returns: M*T_test
    """
    # this is F_t in the paper, (M*T_test)
    M = out_of_sample_signals.shape[1]
    T_test = out_of_sample_signals.shape[2]

    Sigma_epsilon_hat_inv = np.linalg.inv(Sigma_epsilon_hat)
    Sigma_epsilon_hat_inv_T = np.repeat(Sigma_epsilon_hat_inv[np.newaxis, :, :], T_test, axis=0)
    signals_T_times_invSigma_epsilon = np.matmul(out_of_sample_signals.transpose((2, 1, 0)),
                                                 Sigma_epsilon_hat_inv_T).transpose((1, 2, 0)) # M*N*T

    factor_returns = np.sum(signals_T_times_invSigma_epsilon * np.tile(out_of_sample_returns, (M, 1, 1)),1)
    # (M, N, T_test)*(M, N, T_test) by elementwise multiplication

    return factor_returns


def oos_performance(beta_ols, out_of_sample_signals, out_of_sample_returns, Sigma_epsilon_hat):
    """
    This function computes OOS performance of MV portfolios
    :param beta_ols: used estimate of beta
    :param out_of_sample_signals: N*M*T_test
    :param out_of_sample_returns: N*T_test
    :return: oos_mean_return, oos_risk
    """

    if isinstance(beta_ols, list):
        beta_ols = np.array(beta_ols).squeeze().T

    T_test = out_of_sample_returns.shape[1]

    Sigma_epsilon_hat_inv = np.linalg.inv(Sigma_epsilon_hat)
    Sigma_epsilon_hat_inv_T = np.repeat(Sigma_epsilon_hat_inv[np.newaxis, :, :], T_test, axis=0)
    invSigma_epsilon_times_out_of_sample_signals_T = np.matmul(Sigma_epsilon_hat_inv_T, out_of_sample_signals.transpose((2, 0, 1)),
                                                           ).transpose((1, 2, 0)) # N*M*T_test
    oos_mean_return = np.matmul(beta_ols.T, np.tensordot(invSigma_epsilon_times_out_of_sample_signals_T.transpose((1, 0, 2)), # M*N*T_test; out_of_sample_returns: N*T_test; output: M
                                                         out_of_sample_returns, axes=([1, 2], [0, 1])).reshape(-1, 1)) / T_test # N*T

    factor_returns = build_factor_returns(out_of_sample_signals, out_of_sample_returns, Sigma_epsilon_hat)

    # this will be F_tF_t' summed over t, so this is the OOS E[S_t'R_{t+1}R_{t+1}'S_t], (M*M)
    out_of_sample_second_moment = np.matmul(factor_returns, factor_returns.T) / T_test

    if beta_ols.shape[1] > 1:
        oos_risk = np.array([float(np.matmul(np.matmul(beta_ols[:, i].reshape(-1, 1).T, out_of_sample_second_moment), beta_ols[:, i].reshape(-1, 1))) for i in range(beta_ols.shape[1])])# np.diag(np.matmul(np.matmul(beta_ols.T, out_of_sample_second_moment), beta_ols))
    else:
        oos_risk = float(np.matmul(np.matmul(beta_ols.T, out_of_sample_second_moment), beta_ols))

    SRU = oos_mean_return.flatten() / np.sqrt(oos_risk.flatten())
    SRUV = np.sqrt(SRU ** 2 / (1 - SRU ** 2))

    return oos_mean_return, oos_risk, SRU, SRUV


def ols_ridge(in_sample_signals, in_sample_returns, z_list, Sigma_epsilon_hat): # finished
    """
    This is OLS-ridge function
    :param in_sample_signals: in-sample signals with dimension N*M*T
    :param in_sample_returns: in-sample returns with dimension N*T
    :param z: a vector of ridge penalty parameter
    :return: estimated beta of OLS-ridge, beta_ols
    """
    N = in_sample_signals.shape[0]
    M = in_sample_signals.shape[1]
    T = in_sample_signals.shape[2]

    Sigma_epsilon_hat_inv = np.linalg.inv(Sigma_epsilon_hat)
    Sigma_epsilon_hat_inv_T = np.repeat(Sigma_epsilon_hat_inv[np.newaxis, :, :], T, axis=0)
    in_sample_signals_T_times_invSigma_epsilon = np.matmul(in_sample_signals.transpose((2, 1, 0)), Sigma_epsilon_hat_inv_T).transpose((1, 2, 0))
    bar_s = np.tensordot(in_sample_signals_T_times_invSigma_epsilon,
                         in_sample_signals, axes=[(1, 2), (0, 2)]) / (N * T) # M*M

    factor_means = np.tensordot(in_sample_signals_T_times_invSigma_epsilon,
                                in_sample_returns).reshape(-1, 1) / (N * T) # M*1

    betas_ols = [np.matmul(np.linalg.inv(bar_s + z_value * np.eye(M)), factor_means) for z_value in z_list]
    return betas_ols


def betas_MSRR_function(in_sample_signals, in_sample_returns, z, Sigma_epsilon_hat):
    """
    This function estimates beta_MSRR for MSRR
    :param in_sample_signals: in-sample signals with dimension N*M*T
    :param in_sample_returns: in-sample returns with dimension N*T
    :param z: a vector of ridge penalty parameter
    :return: estimated beta of MSRR
    """
    N = in_sample_signals.shape[0]
    M = in_sample_signals.shape[1]
    T = in_sample_signals.shape[2]

    Sigma_epsilon_hat_inv = np.linalg.inv(Sigma_epsilon_hat)
    Sigma_epsilon_hat_inv_T = np.repeat(Sigma_epsilon_hat_inv[np.newaxis, :, :], T, axis=0)
    F_t = np.matmul(np.matmul(in_sample_signals.transpose((2, 1, 0)), Sigma_epsilon_hat_inv_T),
                    np.repeat(in_sample_returns[:, :, np.newaxis], 1, axis=2).transpose((1, 0, 2)))  # (T, M, 1)

    B_T = np.sum(np.matmul(F_t, F_t.transpose((0, 2, 1))), 0) / (N * T)
    # estimate beta_MSRR
    betas_MSRR = [np.matmul(np.linalg.inv(z_value * np.eye(M) + B_T), np.sum(F_t, 0) / (N * T)) for z_value in z]  # M*1
    return betas_MSRR
    # # Check beta_MSRR: beta_MSRR_check should have the same result as beta_MSRR
    # factor_returns_in_sample = build_factor_returns(in_sample_signals, in_sample_returns)  # M*T
    # # B_T is F_tF_t' summed over t/(N*T), the dim is (M*M)
    # B_T = np.matmul(factor_returns_in_sample, factor_returns_in_sample.T) / (N*T)
    # beta_MSRR_check = np.matmul(np.linalg.inv(z_value * np.eye(M) + B_T),
    #                        np.sum(factor_returns_in_sample, 1) / (N * T))  # M*1
    #
    # # Check beta_MSRR 2: similar method as OLS, beta_MSRR_check2 should have the same result as beta_MSRR
    # bar_F = np.tensordot(F_t.transpose((1, 2, 0)),
    #                      F_t.transpose((2, 1, 0)), axes=[(1, 2), (0, 2)]) / (N * T)  # M*M
    # factor_means = np.tensordot(in_sample_signals.transpose((1, 0, 2)),
    #                             in_sample_returns).reshape(-1, 1) / (N * T)  # M*1
    # beta_MSRR_check2 = np.matmul(np.linalg.inv(bar_F + z_value * np.eye(M)), factor_means)  # M*1
    # print([beta_MSRR, beta_MSRR_check, beta_MSRR_check2])


def get_H_SRU_SRUV_b(target_psi, sigma_hat_eigenvalues, psi_eigenvalues):
    """
    This function computes Herfindal, benchmark SRU, benchmark SRUV, and b from target_psi, sigma_hat_eigenvalues, psi_eigenvalues
    :param target_psi:
    :param sigma_hat_eigenvalues: eigenvalues from sigma_hat_function
    :param psi_eigenvalues: true eigenvalues of Psi
    :return:
    """

    M = len(psi_eigenvalues)

    # get Herfindal, benchmark SRU, and benchmark SRUV
    herf = (np.sum(sigma_hat_eigenvalues) ** 2) / (np.sum(sigma_hat_eigenvalues ** 2))
    sru_math = np.sqrt(herf / (2 + herf * (1 + 1 / target_psi)))
    sruv_math = np.sqrt(herf / (2 + herf / target_psi))

    # get b
    b = target_psi * M / (np.sum(psi_eigenvalues) * np.sum(sigma_hat_eigenvalues))

    print(f'comparing two sru: {sru_math, sruv_math}, where herf= {herf}\n '
          f'and targetpsi={target_psi} has led to b={b}')
    return herf, sru_math, sruv_math, b


def initialization(z, N_sim):
    """
    This function generates initialization for function run_new_simulations_for_testing_ols_with_Sigma_epsilon
    :param z: a list of penalty parameters for ridge
    :param N_sim: the number of simulations
    :return:
    """
    # Initialization of m_hat and derivative_of_m_hat for Marcenko-Pastur Theorem
    m_hat = np.zeros(shape=(z.shape[0], N_sim))
    m_Psi_hat = np.zeros(shape=(z.shape[0], N_sim))
    derivative_of_m_hat = np.zeros(shape=(z.shape[0], N_sim))

    # Initialization of ER_Pi_MV_oos for Prop 5
    ER_Pi_MV_oos = np.zeros(shape=(z.shape[0], N_sim))

    # Initialization of ERsq_Pi_MV_oos for Prop 6
    ERsq_Pi_MV_oos = np.zeros(shape=(z.shape[0], N_sim))

    # SRU initialization
    SRU_MV_oos = np.zeros(shape=(z.shape[0], N_sim))
    SRUV_MV_oos = np.zeros(shape=(z.shape[0], N_sim))

    # Initialization for MSRR
    ER_Pi_MSRR_oos = np.zeros(shape=(z.shape[0], N_sim))
    ERsq_Pi_MSRR_oos = np.zeros(shape=(z.shape[0], N_sim))
    SRU_MSRR_oos = np.zeros(shape=(z.shape[0], N_sim))
    SRUV_MSRR_oos = np.zeros(shape=(z.shape[0], N_sim))

    return m_hat, m_Psi_hat, derivative_of_m_hat, ER_Pi_MV_oos, ERsq_Pi_MV_oos, SRU_MV_oos, SRUV_MV_oos, ER_Pi_MSRR_oos, ERsq_Pi_MSRR_oos, SRU_MSRR_oos, SRUV_MSRR_oos


def compute_oos_performance_of_ridge_ols(z, in_sample_signals, in_sample_returns, out_of_sample_signals, out_of_sample_returns, Sigma_epsilon_hat):
    """
    This function computes the out-of-sample performance of OLS-ridge portfolios
    :param z: ridge penalty parameter
    :param N: the number of assets
    :param in_sample_signals: in-sample signals with dimension N*M*T
    :param in_sample_returns: in-sample returns with dimension N*T
    :param out_of_sample_signals: out-of-sample signals with dimension N*M*T_test
    :param out_of_sample_returns: out-of-sample returns with dimension N*T_test
    :param Sigma_epsilon_hat: sigma_hat_function(sigma, sigma_eps)
    :return:
    """
    N = in_sample_returns.shape[0]

    betas_ols = ols_ridge(in_sample_signals, in_sample_returns, z, Sigma_epsilon_hat)
    ols_results = oos_performance(betas_ols, out_of_sample_signals, out_of_sample_returns, Sigma_epsilon_hat)

    # ER_Pi_MV_oos for Proposition 5 test
    ER_Pi_MV_oos_Sim_i = ols_results[0].flatten() # / N
    # ERsq_Pi_MV_oos for Proposition 6 test
    ERsq_Pi_MV_oos_Sim_i = ols_results[1].flatten() #/ (N ** 2)
    # SRU of MV portfolio
    SRU_MV_oos_Sim_i = ols_results[2].flatten()
    SRUV_MV_oos_Sim_i = ols_results[3].flatten()
    return ER_Pi_MV_oos_Sim_i, ERsq_Pi_MV_oos_Sim_i, SRU_MV_oos_Sim_i, SRUV_MV_oos_Sim_i


def compute_oos_performance_of_MSRR(z, in_sample_signals, in_sample_returns, out_of_sample_signals, out_of_sample_returns, Sigma_epsilon_hat):
    """
    This function computes the out-of-sample performance of MSRR portfolios
    :param z: ridge penalty parameter
    :param in_sample_signals: in-sample signals with dimension N*M*T
    :param in_sample_returns: in-sample returns with dimension N*T
    :param out_of_sample_signals: out-of-sample signals with dimension N*M*T_test
    :param out_of_sample_returns: out-of-sample returns with dimension N*T_test
    :param Sigma_epsilon_hat: sigma_hat_function(sigma, sigma_eps)
    :return:
    """
    N = in_sample_returns.shape[0]

    betas_MSRR_hat = betas_MSRR_function(in_sample_signals, in_sample_returns, z, Sigma_epsilon_hat)
    msrr_results = oos_performance(betas_MSRR_hat, out_of_sample_signals, out_of_sample_returns, Sigma_epsilon_hat)

    ER_Pi_MSRR_oos_Sim_i = msrr_results[0].flatten() # / N
    # ERsq_Pi_MV_oos for Proposition 6 test
    ERsq_Pi_MSRR_oos_Sim_i = msrr_results[1].flatten() # / (N ** 2)
    # SRU of MV portfolio
    SRU_MSRR_oos_Sim_i = msrr_results[2].flatten()
    SRUV_MSRR_oos_Sim_i = msrr_results[3].flatten()

    return ER_Pi_MSRR_oos_Sim_i, ERsq_Pi_MSRR_oos_Sim_i, SRU_MSRR_oos_Sim_i, SRUV_MSRR_oos_Sim_i


# def compute_performance_of_MSRR_in_sample(z, in_sample_signals, in_sample_returns, Sigma_epsilon_hat):
#     """
#     This function computes the out-of-sample performance of MSRR portfolios
#     :param z: ridge penalty parameter
#     :param in_sample_signals: in-sample signals with dimension N*M*T
#     :param in_sample_returns: in-sample returns with dimension N*T
#     :param out_of_sample_signals: out-of-sample signals with dimension N*M*T_test
#     :param out_of_sample_returns: out-of-sample returns with dimension N*T_test
#     :param Sigma_epsilon_hat: sigma_hat_function(sigma, sigma_eps)
#     :return:
#     """
#     # N = in_sample_returns.shape[0]
#
#     betas_MSRR_hat = betas_MSRR_function(in_sample_signals, in_sample_returns, z, Sigma_epsilon_hat)
#     msrr_results = oos_performance(betas_MSRR_hat, in_sample_signals, in_sample_returns, Sigma_epsilon_hat)
#
#     ER_Pi_MSRR_oos_Sim_i = msrr_results[0].flatten() # / N
#     # ERsq_Pi_MV_oos for Proposition 6 test
#     ERsq_Pi_MSRR_oos_Sim_i = msrr_results[1].flatten() # / (N ** 2)
#     # SRU of MV portfolio
#     # SRU_MSRR_oos_Sim_i = msrr_results[2].flatten()
#     # SRUV_MSRR_oos_Sim_i = msrr_results[3].flatten()
#
#     return ER_Pi_MSRR_oos_Sim_i, ERsq_Pi_MSRR_oos_Sim_i # , SRU_MSRR_oos_Sim_i, SRUV_MSRR_oos_Sim_i
#

def run_new_simulations_for_testing_ols_with_Sigma_epsilon(N_sim, z, w, N, M, T, T_test, a_Sigma, a_Psi, a_Sigma_eps, target_psi, use_true_sigma_eps):
    """
    This function run simulations for OLS and MSRR testing with Sigma_epsilon
    :param N_sim: number of simulations
    :param z: a vector of ridge penalty parameter
    :param w: a value of ridge penalty parameter for Sigma_epsilon proxy
    :param N: number of assets
    :param M: number of variables
    :param T: training samples
    :param T_test: the number of test samples
    :param a_Sigma: herfindal index for Sigma generation, which controls concentration
    :param a_Psi: herfindal index for Psi generation, which controls concentration
    :param a_Sigma_eps: herfindal index for Sigma_eps generation. When a_Sigma_eps=0, we are back to the old case when Sigma_eps=I
    :param target_psi: target of Psi
    :param use_true_sigma_eps: a boolean variable to use true Sigma_eps or estimated Sigma_eps.
                            When True, use the true Sigma_eps. when False, use the estimate.
    :return:
    """

    # Empirical eigenvalues
    PSI_EIGENVALUES = np.load('../plots_of_eigenvalues/eigenvalues_across_signals_1994-01-31T00:00:00.000000000.npy',
                              allow_pickle=True)

    SIGMA_EIGENVALUES = np.load('../plots_of_eigenvalues/eigenvalues_across_stocks_1994-01-31T00:00:00.000000000.npy',
                                allow_pickle=True)

    SIGMA_EPSILON_EIGENVALUES = np.load('../plots_of_eigenvalues_Sigma_epsilon/eigenvalues_Sigma_epsilon_1994-01-31T00:00:00.000000000.npy',
                                allow_pickle=True)

    # Sigma (covariance of signals across assets) generation
    Sigma, sigma_eigenvalues = generate_Sigma_Psi_with_empirical_eigenvalues(N, a_Sigma, SIGMA_EIGENVALUES)

    # Psi (covariance across signals) generation
    Psi, psi_eigenvalues = generate_Sigma_Psi_with_empirical_eigenvalues(M, a_Psi, PSI_EIGENVALUES)

    # Sigma_eps generation
    Sigma_eps = generate_Sigma_epsilon(N, a_Sigma_eps, SIGMA_EPSILON_EIGENVALUES)

    # Sigma_hat and its eigenvalues
    sigma_hat, sigma_hat_eigenvalues = sigma_hat_function(Sigma, Sigma_eps)

    # get Herfindal, benchmark SRU, benchmark SRUV, and b
    herf, sru_math, sruv_math, b = get_H_SRU_SRUV_b(target_psi, sigma_hat_eigenvalues, psi_eigenvalues)

    # parameters
    c = M / (N * T)
    parameter = (sruv_math, c, N, M, b, T, T_test, N_sim, a_Sigma, a_Psi, target_psi, use_true_sigma_eps, a_Sigma_eps, w)

    # Initialization
    m_hat, m_Psi_hat, derivative_of_m_hat, ER_Pi_MV_oos, ERsq_Pi_MV_oos, SRU_MV_oos, SRUV_MV_oos, ER_Pi_MSRR_oos, ERsq_Pi_MSRR_oos, SRU_MSRR_oos, SRUV_MSRR_oos = initialization(z, N_sim)

    # Run simulations
    for Sim_i in range(N_sim):
        print(str(Sim_i) + '/' + str(N_sim))
        # DGP for each simulations
        beta, in_sample_signals, out_of_sample_signals, in_sample_returns, out_of_sample_returns \
            = generate_data(N, M, T, b, T_test, Sigma, Psi, Sigma_eps)

        # A_T (M*M)
        A_T = np.sum(np.matmul(in_sample_signals.transpose((2, 1, 0)),
                               in_sample_signals.transpose((2, 0, 1))), 0) / (N * T)

        # compute the empirical Stieltjes transform for Marcenko-Pastur Theorem check
        t1 = time.time()
        m_hat[:, Sim_i] = [np.trace(np.linalg.inv(z_value * np.eye(M) + A_T) / M) for z_value in z]
        m_Psi_hat[:, Sim_i] = [map_from_m_to_mPsi(m_hat[i, Sim_i], z[i], M / (N * T)) for i in range(z.shape[0])]
        derivative_of_m_hat[:, Sim_i] = [np.trace(np.linalg.matrix_power(z_value * np.eye(M) + A_T, -2) / M) for z_value in z]

        # Sigma_epsilon estimation
        if use_true_sigma_eps:
            Sigma_epsilon_hat = Sigma_eps
        else:
            Sigma_epsilon_hat = estimate_Sigma_epsilon(in_sample_returns, w)

        # # Check eigenvalues of Sigma_epsilon_hat and Sigma_eps
        # eigenvalues_hat, eigenvectors_hat = np.linalg.eigh(Sigma_epsilon_hat)
        # eigenvalues_Sigma_eps, eigenvectors_Sigma_eps = np.linalg.eigh(Sigma_eps)
        # fig, ax = plt.subplots()
        # plt.scatter(eigenvalues_Sigma_eps, eigenvalues_hat)
        # ax.set_xlabel("eigenvalues of true Sigma_eps")
        # # set y-axis label
        # ax.set_ylabel("eigenvalues of Sigma_eps_hat")
        # plt.title('a_Sigma_eps = %s' % a_Sigma_eps)
        # fig.savefig(os.path.join('/Users/kyzhou/Dropbox/MSRR/Code/Output/Sigma_eps check', f'a_Sigma_eps_{a_Sigma_eps}.png'))

        # OLS
        ER_Pi_MV_oos[:, Sim_i], ERsq_Pi_MV_oos[:, Sim_i], SRU_MV_oos[:, Sim_i], SRUV_MV_oos[:, Sim_i] = \
            compute_oos_performance_of_ridge_ols(z, in_sample_signals, in_sample_returns, out_of_sample_signals, out_of_sample_returns, Sigma_epsilon_hat)

        # MSRR
        ER_Pi_MSRR_oos[:, Sim_i], ERsq_Pi_MSRR_oos[:, Sim_i], SRU_MSRR_oos[:, Sim_i], SRUV_MSRR_oos[:, Sim_i] = \
            compute_oos_performance_of_MSRR(z, in_sample_signals, in_sample_returns, out_of_sample_signals, out_of_sample_returns, Sigma_epsilon_hat)

        t2 = time.time()
        print(f'loop over z took {t2 - t1}')

    return m_hat.mean(1), derivative_of_m_hat.mean(1), m_Psi_hat.mean(1), ER_Pi_MV_oos.mean(1), \
           ERsq_Pi_MV_oos.mean(1), SRU_MV_oos.mean(1), SRUV_MV_oos.mean(1), \
           ER_Pi_MSRR_oos.mean(1), ERsq_Pi_MSRR_oos.mean(1), SRU_MSRR_oos.mean(1), SRUV_MSRR_oos.mean(1), \
           parameter


def initialization_wlist(z, N_sim, w):
    """
    This function generates initialization for function run_new_simulations_for_testing_ols_with_Sigma_epsilon_wlist
    :param z: a list of penalty parameters for ridge
    :param N_sim: the number of simulations
    :param w: a list of ridge penalty parameter for Sigma_epsilon proxy
    :return:
    """

    # Initialization of m_hat and derivative_of_m_hat for Marcenko-Pastur Theorem
    m_hat = np.zeros(shape=(z.shape[0], N_sim))
    m_Psi_hat = np.zeros(shape=(z.shape[0], N_sim))
    derivative_of_m_hat = np.zeros(shape=(z.shape[0], N_sim))

    # MV
    ER_Pi_MV_oos = np.zeros(shape=(z.shape[0], N_sim, len(w)))
    ERsq_Pi_MV_oos = np.zeros(shape=(z.shape[0], N_sim, len(w)))
    SRU_MV_oos = np.zeros(shape=(z.shape[0], N_sim, len(w)))
    SRUV_MV_oos = np.zeros(shape=(z.shape[0], N_sim, len(w)))

    # MSRR
    ER_Pi_MSRR_oos = np.zeros(shape=(z.shape[0], N_sim, len(w)))
    ERsq_Pi_MSRR_oos = np.zeros(shape=(z.shape[0], N_sim, len(w)))
    SRU_MSRR_oos = np.zeros(shape=(z.shape[0], N_sim, len(w)))
    SRUV_MSRR_oos = np.zeros(shape=(z.shape[0], N_sim, len(w)))
    return m_hat, m_Psi_hat, derivative_of_m_hat, ER_Pi_MV_oos, ERsq_Pi_MV_oos, SRU_MV_oos, SRUV_MV_oos, ER_Pi_MSRR_oos, ERsq_Pi_MSRR_oos, SRU_MSRR_oos, SRUV_MSRR_oos




def run_new_simulations_for_testing_ols_with_Sigma_epsilon_wlist(N_sim, z, w, N, M, T, T_test, a_Sigma, a_Psi, a_Sigma_eps, target_psi, use_true_sigma_eps):
    """
    This function run simulations for OLS and MSRR testing with Sigma_epsilon
    :param N_sim: number of simulations
    :param z: a vector of ridge penalty parameter
    :param w: a vector of ridge penalty parameter for Sigma_epsilon proxy
    :param N: number of assets
    :param M: number of variables
    :param T: training samples
    :param T_test: the number of test samples
    :param a_Sigma: herfindal index for Sigma generation, which controls concentration
    :param a_Psi: herfindal index for Psi generation, which controls concentration
    :param a_Sigma_eps: herfindal index for Sigma_eps generation. When a_Sigma_eps=0, we are back to the old case when Sigma_eps=I
    :param target_psi: target of Psi
    :param use_true_sigma_eps: a boolean variable to use true Sigma_eps or estimated Sigma_eps.
                            When True, use the true Sigma_eps. when False, use the estimate.
    :return:
    """

    # Empirical eigenvalues
    PSI_EIGENVALUES = np.load('../plots_of_eigenvalues/eigenvalues_across_signals_1994-01-31T00:00:00.000000000.npy',
                              allow_pickle=True)

    SIGMA_EIGENVALUES = np.load('../plots_of_eigenvalues/eigenvalues_across_stocks_1994-01-31T00:00:00.000000000.npy',
                                allow_pickle=True)

    SIGMA_EPSILON_EIGENVALUES = np.load('../plots_of_eigenvalues_Sigma_epsilon/eigenvalues_Sigma_epsilon_1994-01-31T00:00:00.000000000.npy',
                                allow_pickle=True)

    # Sigma (covariance of signals across assets) generation
    Sigma, sigma_eigenvalues = generate_Sigma_Psi_with_empirical_eigenvalues(N, a_Sigma, SIGMA_EIGENVALUES)

    # Psi (covariance across signals) generation
    Psi, psi_eigenvalues = generate_Sigma_Psi_with_empirical_eigenvalues(M, a_Psi, PSI_EIGENVALUES)

    # Sigma_eps generation
    Sigma_eps = generate_Sigma_epsilon(N, a_Sigma_eps) # , SIGMA_EPSILON_EIGENVALUES)

    # Sigma_hat and its eigenvalues
    sigma_hat, sigma_hat_eigenvalues = sigma_hat_function(Sigma, Sigma_eps)

    # get Herfindal, benchmark SRU, benchmark SRUV, and b
    herf, sru_math, sruv_math, b = get_H_SRU_SRUV_b(target_psi, sigma_hat_eigenvalues, psi_eigenvalues)

    # parameters
    c = M / (N * T)
    parameter = (sruv_math, c, N, M, b, T, T_test, N_sim, a_Sigma, a_Psi, target_psi, use_true_sigma_eps, a_Sigma_eps, w)

    # Initialization
    if use_true_sigma_eps:
        m_hat, m_Psi_hat, derivative_of_m_hat, ER_Pi_MV_oos, ERsq_Pi_MV_oos, SRU_MV_oos, SRUV_MV_oos, ER_Pi_MSRR_oos, ERsq_Pi_MSRR_oos, SRU_MSRR_oos, SRUV_MSRR_oos = initialization(z, N_sim)
    else:
        m_hat, m_Psi_hat, derivative_of_m_hat, ER_Pi_MV_oos, ERsq_Pi_MV_oos, SRU_MV_oos, SRUV_MV_oos, ER_Pi_MSRR_oos, ERsq_Pi_MSRR_oos, SRU_MSRR_oos, SRUV_MSRR_oos = initialization_wlist(z, N_sim, w)

    # Run simulations
    for Sim_i in range(N_sim):
        print(str(Sim_i) + '/' + str(N_sim))
        # DGP for each simulations
        beta, in_sample_signals, out_of_sample_signals, in_sample_returns, out_of_sample_returns, epsilon \
            = generate_data(N, M, T, b, T_test, Sigma, Psi, Sigma_eps)

        # A_T (M*M)
        A_T = np.sum(np.matmul(in_sample_signals.transpose((2, 1, 0)),
                               in_sample_signals.transpose((2, 0, 1))), 0) / (N * T)

        # compute the empirical Stieltjes transform for Marcenko-Pastur Theorem check
        t1 = time.time()
        m_hat[:, Sim_i] = [np.trace(np.linalg.inv(z_value * np.eye(M) + A_T) / M) for z_value in z]
        m_Psi_hat[:, Sim_i] = [map_from_m_to_mPsi(m_hat[i, Sim_i], z[i], M / (N * T)) for i in range(z.shape[0])]
        derivative_of_m_hat[:, Sim_i] = [np.trace(np.linalg.matrix_power(z_value * np.eye(M) + A_T, -2) / M) for z_value in z]

        # Sigma_epsilon estimation
        if use_true_sigma_eps:
            Sigma_epsilon_hat = Sigma_eps
            # OLS
            ER_Pi_MV_oos[:, Sim_i], ERsq_Pi_MV_oos[:, Sim_i], SRU_MV_oos[:, Sim_i], SRUV_MV_oos[:, Sim_i] = \
                compute_oos_performance_of_ridge_ols(z, in_sample_signals, in_sample_returns, out_of_sample_signals,
                                                     out_of_sample_returns, Sigma_epsilon_hat)
            # MSRR
            ER_Pi_MSRR_oos[:, Sim_i], ERsq_Pi_MSRR_oos[:, Sim_i], SRU_MSRR_oos[:, Sim_i], SRUV_MSRR_oos[:, Sim_i] = \
                compute_oos_performance_of_MSRR(z, in_sample_signals, in_sample_returns, out_of_sample_signals,
                                                out_of_sample_returns, Sigma_epsilon_hat)
        else:
            for idx_w in range(len(w)):
                w_i = w[idx_w]
                Sigma_epsilon_hat = estimate_Sigma_epsilon(in_sample_returns, w_i)

                # OLS
                ER_Pi_MV_oos[:, Sim_i, idx_w], ERsq_Pi_MV_oos[:, Sim_i, idx_w], SRU_MV_oos[:, Sim_i, idx_w], SRUV_MV_oos[:, Sim_i, idx_w] = \
                    compute_oos_performance_of_ridge_ols(z, in_sample_signals, in_sample_returns, out_of_sample_signals,
                                                         out_of_sample_returns, Sigma_epsilon_hat)
                # MSRR
                ER_Pi_MSRR_oos[:, Sim_i, idx_w], ERsq_Pi_MSRR_oos[:, Sim_i, idx_w], SRU_MSRR_oos[:, Sim_i, idx_w], SRUV_MSRR_oos[:, Sim_i, idx_w] = \
                    compute_oos_performance_of_MSRR(z, in_sample_signals, in_sample_returns, out_of_sample_signals,
                                                    out_of_sample_returns, Sigma_epsilon_hat)
        t2 = time.time()
        print(f'loop over z took {t2 - t1}')

    return m_hat.mean(1), derivative_of_m_hat.mean(1), m_Psi_hat.mean(1), ER_Pi_MV_oos.mean(1), \
           ERsq_Pi_MV_oos.mean(1), SRU_MV_oos.mean(1), SRUV_MV_oos.mean(1), \
           ER_Pi_MSRR_oos.mean(1), ERsq_Pi_MSRR_oos.mean(1), SRU_MSRR_oos.mean(1), SRUV_MSRR_oos.mean(1), \
           parameter



def run_new_simulations_for_testing_ols_with_Sigma_epsilon_wlist_cov(N_sim, z, w, N, M, T, T_test, Sigma, Psi, Sigma_eps, b, use_true_sigma_eps):
    """
    This function run simulations for OLS and MSRR testing with Sigma_epsilon
    :param N_sim: number of simulations
    :param z: a vector of ridge penalty parameter
    :param w: a vector of ridge penalty parameter for Sigma_epsilon proxy
    :param N: number of assets
    :param M: number of variables
    :param T: training samples
    :param T_test: the number of test samples
    :param a_Sigma: herfindal index for Sigma generation, which controls concentration
    :param a_Psi: herfindal index for Psi generation, which controls concentration
    :param a_Sigma_eps: herfindal index for Sigma_eps generation. When a_Sigma_eps=0, we are back to the old case when Sigma_eps=I
    :param target_psi: target of Psi
    :param use_true_sigma_eps: a boolean variable to use true Sigma_eps or estimated Sigma_eps.
                            When True, use the true Sigma_eps. when False, use the estimate.
    :return:
    """

    # # Sigma (covariance of signals across assets) generation
    # Sigma, sigma_eigenvalues = generate_Sigma_with_empirical_eigenvalues(N, a_Sigma)
    #
    # # Psi (covariance across signals) generation
    # Psi, psi_eigenvalues = generate_Psi_with_empirical_eigenvalues(M, a_Psi)
    #
    # # Sigma_eps generation
    # Sigma_eps = generate_Sigma_epsilon(N, a_Sigma_eps)

    # Sigma_hat and its eigenvalues
    sigma_hat, sigma_hat_eigenvalues = sigma_hat_function(Sigma, Sigma_eps)

    # # get Herfindal, benchmark SRU, benchmark SRUV, and b
    # herf, sru_math, sruv_math, b = get_H_SRU_SRUV_b(target_psi, sigma_hat_eigenvalues, psi_eigenvalues)

    # parameters
    c = M / (N * T)
    parameter = (c, N, M, b, T, T_test, N_sim, use_true_sigma_eps)

    # Initialization
    if use_true_sigma_eps:
        m_hat, m_Psi_hat, derivative_of_m_hat, ER_Pi_MV_oos, ERsq_Pi_MV_oos, SRU_MV_oos, SRUV_MV_oos, ER_Pi_MSRR_oos, ERsq_Pi_MSRR_oos, SRU_MSRR_oos, SRUV_MSRR_oos = initialization(z, N_sim)
    else:
        m_hat, m_Psi_hat, derivative_of_m_hat, ER_Pi_MV_oos, ERsq_Pi_MV_oos, SRU_MV_oos, SRUV_MV_oos, ER_Pi_MSRR_oos, ERsq_Pi_MSRR_oos, SRU_MSRR_oos, SRUV_MSRR_oos = initialization_wlist(z, N_sim, w)

    # Run simulations
    for Sim_i in range(N_sim):
        print(str(Sim_i) + '/' + str(N_sim))
        # DGP for each simulations
        beta, in_sample_signals, out_of_sample_signals, in_sample_returns, out_of_sample_returns, epsilon \
            = generate_data(N, M, T, b, T_test, Sigma, Psi, Sigma_eps)

        # A_T (M*M)
        A_T = np.sum(np.matmul(in_sample_signals.transpose((2, 1, 0)),
                               in_sample_signals.transpose((2, 0, 1))), 0) / (N * T)

        # compute the empirical Stieltjes transform for Marcenko-Pastur Theorem check
        t1 = time.time()
        m_hat[:, Sim_i] = [np.trace(np.linalg.inv(z_value * np.eye(M) + A_T) / M) for z_value in z]
        m_Psi_hat[:, Sim_i] = [map_from_m_to_mPsi(m_hat[i, Sim_i], z[i], M / (N * T)) for i in range(z.shape[0])]
        derivative_of_m_hat[:, Sim_i] = [np.trace(np.linalg.matrix_power(z_value * np.eye(M) + A_T, -2) / M) for z_value in z]

        # Sigma_epsilon estimation
        if use_true_sigma_eps:
            Sigma_epsilon_hat = Sigma_eps
            # OLS
            ER_Pi_MV_oos[:, Sim_i], ERsq_Pi_MV_oos[:, Sim_i], SRU_MV_oos[:, Sim_i], SRUV_MV_oos[:, Sim_i] = \
                compute_oos_performance_of_ridge_ols(z, in_sample_signals, in_sample_returns, out_of_sample_signals,
                                                     out_of_sample_returns, Sigma_epsilon_hat)
            # MSRR
            ER_Pi_MSRR_oos[:, Sim_i], ERsq_Pi_MSRR_oos[:, Sim_i], SRU_MSRR_oos[:, Sim_i], SRUV_MSRR_oos[:, Sim_i] = \
                compute_oos_performance_of_MSRR(z, in_sample_signals, in_sample_returns, out_of_sample_signals,
                                                out_of_sample_returns, Sigma_epsilon_hat)
        else:
            for idx_w in range(len(w)):
                w_i = w[idx_w]
                Sigma_epsilon_hat = estimate_Sigma_epsilon(in_sample_returns, w_i)

                # OLS
                ER_Pi_MV_oos[:, Sim_i, idx_w], ERsq_Pi_MV_oos[:, Sim_i, idx_w], SRU_MV_oos[:, Sim_i, idx_w], SRUV_MV_oos[:, Sim_i, idx_w] = \
                    compute_oos_performance_of_ridge_ols(z, in_sample_signals, in_sample_returns, out_of_sample_signals,
                                                         out_of_sample_returns, Sigma_epsilon_hat)
                # MSRR
                ER_Pi_MSRR_oos[:, Sim_i, idx_w], ERsq_Pi_MSRR_oos[:, Sim_i, idx_w], SRU_MSRR_oos[:, Sim_i, idx_w], SRUV_MSRR_oos[:, Sim_i, idx_w] = \
                    compute_oos_performance_of_MSRR(z, in_sample_signals, in_sample_returns, out_of_sample_signals,
                                                    out_of_sample_returns, Sigma_epsilon_hat)
        t2 = time.time()
        print(f'loop over z took {t2 - t1}')

    return m_hat.mean(1), derivative_of_m_hat.mean(1), m_Psi_hat.mean(1), ER_Pi_MV_oos.mean(1), \
           ERsq_Pi_MV_oos.mean(1), SRU_MV_oos.mean(1), SRUV_MV_oos.mean(1), \
           ER_Pi_MSRR_oos.mean(1), ERsq_Pi_MSRR_oos.mean(1), SRU_MSRR_oos.mean(1), SRUV_MSRR_oos.mean(1), \
           parameter


def initialization_Gamma(z, N_sim):
    """
    This function generates initialization for Gamma in Lemma 7
    :param z: a list of penalty parameters for ridge
    :param N_sim: the number of simulations
    :return:
    """
    Gamma0_T_check = np.zeros(shape=(len(z), N_sim))
    Gamma1_T_check = np.zeros(shape=(len(z), N_sim))
    Gamma2_T_check = np.zeros(shape=(len(z), N_sim))
    Gamma3_T_check = np.zeros(shape=(len(z), N_sim))
    Gamma4_T_check = np.zeros(shape=(len(z), N_sim))
    return Gamma0_T_check, Gamma1_T_check, Gamma2_T_check, Gamma3_T_check, Gamma4_T_check


def performance_Appendix_A_theoretical(beta_hat, Sigma_epsilon_hat, Psi, beta, b, M):
    '''
    This function computes returns and risks according to Appendix A
    :param beta_hat:
    :param Sigma_epsilon_hat:
    :param Psi:
    :param beta:
    :return:
    '''
    ER_oos = np.matmul(beta_hat.T, np.matmul(Psi, beta)) * np.trace(Sigma_epsilon_hat)
    ERsq_const = ((np.trace(Sigma_epsilon_hat) ** 2 + np.trace(Sigma_epsilon_hat ** 2))\
                  * np.matmul(np.matmul(Psi, beta),np.matmul(beta.T, Psi))\
                + Psi * (np.trace(Sigma_epsilon_hat) + b / M * np.trace(Psi) * np.trace(Sigma_epsilon_hat ** 2)))
    ERsq_oos = np.matmul(beta_hat.T, np.matmul(ERsq_const, beta_hat))
    return ER_oos, ERsq_oos


def run_new_simulations_for_testing_ols_with_Sigma_epsilon_wlist_cov_AppendixA(N_sim, z, w, N, M, T, K, Sigma, Psi, Sigma_eps, b, use_true_sigma_eps, code_path, special_str):
    """
    This function run simulations for OLS and MSRR testing with Sigma_epsilon according to Appendix A
    :param N_sim: number of simulations
    :param z: a vector of ridge penalty parameter
    :param w: a vector of ridge penalty parameter for Sigma_epsilon proxy
    :param N: number of assets
    :param M: number of variables
    :param T: training samples
    :param a_Sigma: herfindal index for Sigma generation, which controls concentration
    :param a_Psi: herfindal index for Psi generation, which controls concentration
    :param a_Sigma_eps: herfindal index for Sigma_eps generation. When a_Sigma_eps=0, we are back to the old case when Sigma_eps=I
    :param target_psi: target of Psi
    :param use_true_sigma_eps: a boolean variable to use true Sigma_eps or estimated Sigma_eps.
                            When True, use the true Sigma_eps. when False, use the estimate.
    :return:
    """

    # sigma_hat, sigma_hat_eigenvalues = sigma_hat_function(Sigma, Sigma_eps)

    # parameters
    c = M / (N * T)
    c_M = M/T
    parameter = (c, N, M, b, T, N_sim, use_true_sigma_eps)

    # Initialization
    if use_true_sigma_eps:
        m_hat, m_Psi_hat, derivative_of_m_hat, ER_Pi_MV_oos, ERsq_Pi_MV_oos, SRU_MV_oos, SRUV_MV_oos, ER_Pi_MSRR_oos, ERsq_Pi_MSRR_oos, SRU_MSRR_oos, SRUV_MSRR_oos = initialization(z, N_sim)
    else:
        m_hat, m_Psi_hat, derivative_of_m_hat, ER_Pi_MV_oos, ERsq_Pi_MV_oos, SRU_MV_oos, SRUV_MV_oos, ER_Pi_MSRR_oos, ERsq_Pi_MSRR_oos, SRU_MSRR_oos, SRUV_MSRR_oos = initialization_wlist(z, N_sim, w)

    # Initialize Gamma in Lemma 7
    Gamma0_T_check, Gamma1_T_check, Gamma2_T_check, Gamma3_T_check, Gamma4_T_check = initialization_Gamma(z, N_sim)

    # Run simulations
    for Sim_i in range(N_sim):
        print(str(Sim_i) + '/' + str(N_sim))
        # DGP for each simulations
        DGS_start_time = time.time()
        beta, in_sample_signals, out_of_sample_signals, in_sample_returns, out_of_sample_returns, epsilon \
            = generate_data(N, M, T, b, 0, Sigma, Psi, Sigma_eps)
        print("--- DGS %s seconds ---" % (time.time() - DGS_start_time))

        # A_T (M*M)
        temp_time = time.time()
        A_T = np.sum(np.matmul(in_sample_signals.transpose((2, 1, 0)),
                               in_sample_signals.transpose((2, 0, 1))), 0) / (N * T)
        print("--- A_T %s seconds ---" % (time.time() - temp_time))

        # compute the empirical Stieltjes transform for Marcenko-Pastur Theorem check
        t1 = time.time()
        m_hat[:, Sim_i] = [np.trace(np.linalg.inv(z_value * np.eye(M) + A_T) / M) for z_value in z]
        m_Psi_hat[:, Sim_i] = [map_from_m_to_mPsi(m_hat[i, Sim_i], z[i], M / (N * T)) for i in range(z.shape[0])]
        derivative_of_m_hat[:, Sim_i] = [np.trace(np.linalg.matrix_power(z_value * np.eye(M) + A_T, -2) / M) for z_value in z]

        # Sigma_epsilon estimation
        Sigma_epsilon_hat = Sigma_eps
        ERsq_const = ((np.trace(Sigma_epsilon_hat) ** 2 + np.trace(Sigma_epsilon_hat ** 2))\
                      * np.matmul(np.matmul(Psi, beta),np.matmul(beta.T, Psi))\
                    + Psi * (np.trace(Sigma_epsilon_hat) + b / M * np.trace(Psi) * np.trace(Sigma_epsilon_hat ** 2)))

        # MV portfolios
        betas_ols_hat = np.array(ols_ridge(in_sample_signals, in_sample_returns, z, Sigma_epsilon_hat))
        # ER_Pi_MV_oos[:, Sim_i], ERsq_Pi_MV_oos[:, Sim_i] = performance_Appendix_A_theoretical(betas_ols_hat, Sigma_epsilon_hat, Psi, beta, b, M)
        ER_Pi_MV_oos[:, Sim_i] = np.squeeze(np.matmul(betas_ols_hat.transpose((0,2,1)), np.matmul(Psi, beta)) * np.trace(Sigma_epsilon_hat))
        ERsq_Pi_MV_oos[:, Sim_i] = np.squeeze(np.matmul(betas_ols_hat.transpose((0,2,1)), np.matmul(ERsq_const, betas_ols_hat)))

        # MSRR portfolio
        betas_MSRR_hat = np.array(betas_MSRR_function(in_sample_signals, in_sample_returns, z, Sigma_epsilon_hat))
        # ER_Pi_MSRR_oos[:, Sim_i], ERsq_Pi_MSRR_oos[:, Sim_i] = performance_Appendix_A_theoretical(betas_MSRR_hat,Sigma_epsilon_hat, Psi,beta, b, M)
        ER_Pi_MSRR_oos[:, Sim_i] = np.squeeze(np.matmul(betas_MSRR_hat.transpose((0,2,1)), np.matmul(Psi, beta)) * np.trace(Sigma_epsilon_hat))
        ERsq_Pi_MSRR_oos[:, Sim_i] = np.squeeze(np.matmul(betas_MSRR_hat.transpose((0,2,1)), np.matmul(ERsq_const, betas_MSRR_hat)))

        # from function betas_MSRR_function:
        Sigma_epsilon_hat_inv = np.linalg.inv(Sigma_epsilon_hat)
        Sigma_epsilon_hat_inv_T = np.repeat(Sigma_epsilon_hat_inv[np.newaxis, :, :], T, axis=0)

        F_t_start_time = time.time()
        F_t = np.matmul(np.matmul(in_sample_signals.transpose((2, 1, 0)), Sigma_epsilon_hat_inv_T),
                        np.repeat(in_sample_returns[:, :, np.newaxis], 1, axis=2).transpose((1, 0, 2)))  # (T, M, 1)
        print("--- F_t %s seconds ---" % (time.time() - F_t_start_time))

        B_T_start_time = time.time()
        B_T = np.sum(np.matmul(F_t, F_t.transpose((0, 2, 1))), 0) / (N * T)  # (M,M)
        print("--- B_T %s seconds ---" % (time.time() - B_T_start_time))

        I = np.eye(M)

        gamma_start_time = time.time()
        Gamma0_T_check[:, Sim_i]  = np.array([np.matmul(beta.T, np.matmul(np.linalg.inv(z[z_idx] * I + B_T), beta))[0, 0] for
                                   z_idx in range(len(z))])
        print("--- Gamma_0 %s seconds ---" % (time.time() - gamma_start_time))

        # Gamma1
        gamma_start_time = time.time()
        Gamma1_T_check[:, Sim_i] = np.squeeze([np.matmul(beta.T, np.matmul(np.matmul(Psi, np.linalg.inv(z[z_idx] * I + B_T)), beta)) for
                    z_idx in range(len(z))])
        print("--- Gamma_1 %s seconds ---" % (time.time() - gamma_start_time))

        # Gamma2
        gamma_start_time = time.time()
        Gamma2_T_check[:, Sim_i] = np.squeeze([
            np.matmul(beta.T, np.matmul(np.matmul(Psi**2, np.linalg.inv(z[z_idx] * I + B_T)), beta))
            for z_idx in
            range(len(z))])
        print("--- Gamma_2 %s seconds ---" % (time.time() - gamma_start_time))

        # Gamma3
        gamma_start_time = time.time()
        Gamma3_T_check[:, Sim_i] = np.array([1 / M * np.trace(np.matmul(np.matmul(Psi, np.linalg.inv(z[z_idx] * I + B_T)),
                                               np.matmul(Psi, np.linalg.inv(z[z_idx] * I + B_T)))) for z_idx in
                    range(len(z))])
        # Gamma3_T_check[:, Sim_i] = np.squeeze([np.matmul(beta.T, np.matmul(np.matmul(Psi**3, np.linalg.inv(z[z_idx] * I + B_T)), beta)) for
        #             z_idx in range(len(z))])
        print("--- Gamma_3 %s seconds ---" % (time.time() - gamma_start_time))

        # Gamma4
        gamma_start_time = time.time()
        Gamma4_T_expectation = [np.matmul(np.linalg.inv(z[z_idx] * I + B_T), np.matmul(Psi, np.linalg.inv(z[z_idx] * I + B_T))) for
            z_idx in range(len(z))]
        betaT_Psi = np.matmul(beta.T, Psi)
        Psi_beta = np.matmul(Psi, beta)
        Gamma4_T_check[:, Sim_i] = np.array([np.squeeze(np.matmul(betaT_Psi, np.matmul(Gamma4_T_expectation[z_idx], Psi_beta))) for z_idx in
                    range(len(z))])
        # Gamma4_T_check[:, Sim_i] = np.squeeze(
        #     [np.matmul(beta.T, np.matmul(np.matmul(Psi ** 4, np.linalg.inv(z[z_idx] * I + B_T)), beta)) for
        #      z_idx in range(len(z))])
        print("--- Gamma_4 %s seconds ---" % (time.time() - gamma_start_time))

        save_path = os.path.join(code_path, 'Save_Data')
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        save_path = os.path.join(save_path, 'N%s_M%s_T%s'%(N,M,T))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        with open(save_path + '/save_N%s_M%s_T%s_K%s_Sim%s_%s.pkl' % (N, M, T, K, Sim_i, special_str), 'wb') as f:
            pickle.dump(
                [beta, in_sample_signals, in_sample_returns, epsilon, F_t, B_T, A_T], f)
                    # with open(save_path + '/save_N%s_M%s_T%s_Sim%s.pkl' % (N, M, T, Sim_i), 'rb') as f:  # Python 3: open(..., 'rb')
            #     [beta, in_sample_signals, in_sample_returns, F_t, Gamma1_T_check, Gamma2_T_check, Gamma3_T_check,
            #     Gamma4_T_check] = pickle.load(f)

        t2 = time.time()
        print(f'loop over z took {t2 - t1}')

    return m_hat.mean(1), derivative_of_m_hat.mean(1), m_Psi_hat.mean(1), ER_Pi_MV_oos.mean(1), \
           ERsq_Pi_MV_oos.mean(1), ER_Pi_MSRR_oos.mean(1), ERsq_Pi_MSRR_oos.mean(1), \
           parameter, np.squeeze(Gamma0_T_check.mean(1)), np.squeeze(Gamma1_T_check.mean(1)), \
           np.squeeze(Gamma2_T_check.mean(1)), np.squeeze(Gamma3_T_check.mean(1)), np.squeeze(Gamma4_T_check.mean(1))

