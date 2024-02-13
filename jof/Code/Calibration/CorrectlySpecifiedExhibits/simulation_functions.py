# This file contains functions for running out-of-sample simulations.
import numpy as np
from auxilliary_functions import matrix_square_root
from Eigenvalue_functions import map_from_m_to_mPsi
from expectation_functions import benchmark_SRU
import time


def generate_basic_data(N, M, T, b, T_test, Sigma, Psi):
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
    epsilon = np.random.randn(N, T + T_test)
    returns = np.tensordot(S, beta, axes=(1, 0)).squeeze() + epsilon  # (N, T + T_test)
    return beta, S, epsilon, returns


def generate_data(N, M, T, b, T_test, Sigma, Psi):
    """
    :param N: number of assets
    :param M: number of variables
    :param T: training samples
    :param b: the level of beta, b = ||beta||^2
    :param T_test: the number of test samples
    :param K_perc: percentile of non-zero elements in the diagnal of Sigma
    :param Sigma: covariance of signals across assets for S generation
    :param Psi: covariance across signals for S generation
    :return: beta, in_sample_signals, out_of_sample_signals, in_sample_returns, out_of_sample_returns
    """

    beta, S, epsilon, returns = generate_basic_data(N, M, T, b, T_test, Sigma, Psi)

    # in-sample
    in_sample_signals = S[:, :, :T]
    in_sample_returns = returns[:, :T]

    # OOS
    out_of_sample_signals = S[:, :, T:]
    out_of_sample_returns = returns[:, T:]
    return beta, in_sample_signals, out_of_sample_signals, in_sample_returns, out_of_sample_returns


def generate_Sigma_Psi(k, N, a, sigma_squared):
    """
    This function generates Sigma or Psi.
    For Sigma generation, the second parameter is N. Sigma = (k_Sigma, N, a, sigma_squared)
    For Psi generation, the second parameter is M. Psi = (k, M, a, sigma_squared)
    :param k: the fraction of factors in N or M. k \in [0, 1]
    :param N: Number of assets (N) or number of signals (M).
    :param a: herfindal index, which controls concentration. A large a corresponds to very high concentration of eigenvalues. a > 0
    :param sigma_squared: Signal noise variance. sigma_squared > 0.
    :return:
    """

    # number of common factors in signals
    K = max(np.int(k * N),1) + 1

    # Sigma initialization
    Sigma = np.identity(N)

    # q(a, K) = 1/sum_{i = 1}^K i^a
    q = 1 / np.sum(np.power(np.arange(1, K),a))

    # diagonal elements
    diag_elements = np.concatenate([np.ones(N - K+1) * sigma_squared * q, q * (np.arange(1, K) ** a)])
    np.fill_diagonal(Sigma, diag_elements)
    return Sigma


def build_factor_returns(out_of_sample_signals, out_of_sample_returns):
    """
    This just does S_t'R_{t+1} but in a smart way
    :param out_of_sample_signals: out-of-sample signals with dimension N*M*T_test
    :param out_of_sample_returns: out-of-sample returns with dimension N*T_test
    :return: factor_returns: M*T_test
    """
    # this is F_t in the paper, (M*T_test)
    M = out_of_sample_signals.shape[1]
    factor_returns = np.sum(out_of_sample_signals.transpose((1, 0, 2)) * np.tile(out_of_sample_returns, (M, 1, 1)),
                            1)  # (M, N, T_test)*(M, N, T_test) by elementwise multiplication
    return factor_returns


def oos_performance(beta_ols, out_of_sample_signals, out_of_sample_returns):
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
    oos_mean_return = np.matmul(beta_ols.T, np.tensordot(out_of_sample_signals.transpose((1, 0, 2)), # M*N*T_test; out_of_sample_returns: N*T_test; output: M
                                                         out_of_sample_returns, axes=([1, 2], [0, 1])).reshape(-1, 1)) / T_test # N*T

    factor_returns = build_factor_returns(out_of_sample_signals, out_of_sample_returns)

    # this will be F_tF_t' summed over t, so this is the OOS E[S_t'R_{t+1}R_{t+1}'S_t], (M*M)
    out_of_sample_second_moment = np.matmul(factor_returns, factor_returns.T) / T_test

    if beta_ols.shape[1] > 1:
        oos_risk = np.array([float(np.matmul(np.matmul(beta_ols[:, i].reshape(-1, 1).T, out_of_sample_second_moment), beta_ols[:, i].reshape(-1, 1))) for i in range(beta_ols.shape[1])])# np.diag(np.matmul(np.matmul(beta_ols.T, out_of_sample_second_moment), beta_ols))
    else:
        oos_risk = float(np.matmul(np.matmul(beta_ols.T, out_of_sample_second_moment), beta_ols))

    SRU = oos_mean_return.flatten() / np.sqrt(oos_risk.flatten())
    SRUV = np.sqrt(SRU ** 2 / (1 - SRU ** 2))

    return oos_mean_return, oos_risk, SRU, SRUV


def ols_ridge(in_sample_signals, in_sample_returns, z_list): # finished
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

    bar_s = np.tensordot(in_sample_signals.transpose((1, 0, 2)),
                         in_sample_signals, axes=[(1, 2), (0, 2)]) / (N * T) # M*M
    factor_means = np.tensordot(in_sample_signals.transpose((1, 0, 2)),
                                in_sample_returns).reshape(-1, 1) / (N * T) # M*1
    betas_ols = [np.matmul(np.linalg.inv(bar_s + z_value * np.eye(M)), factor_means) for z_value in z_list]
    return betas_ols


def compute_oos_performance_of_ridge_ols_for_fixed_z(z_value, in_sample_signals,
                                                     in_sample_returns, out_of_sample_signals, out_of_sample_returns):
    """
    This function computes the out-of-sample performance of OLS-ridge portfolios
    :param z_value: a number of ridge penalty parameter, note that it's not a vector
    :param in_sample_signals: in-sample signals with dimension N*M*T
    :param in_sample_returns: in-sample returns with dimension N*T
    :param out_of_sample_signals: out-of-sample signals with dimension N*M*T_test
    :param out_of_sample_returns: out-of-sample returns with dimension N*T_test
    :return: returns and risks (squared returns) of the OLS-ridge portfolios
    """
    beta_ols = ols_ridge(in_sample_signals, in_sample_returns, z_value)
    loss_return_ols, loss_risk_ols, SRU_ols = oos_performance(beta_ols, out_of_sample_signals, out_of_sample_returns)
    return loss_return_ols, loss_risk_ols, SRU_ols


def betas_MSRR_function(in_sample_signals, in_sample_returns, z):
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

    F_t = np.matmul(in_sample_signals.transpose((2, 1, 0)),
                    np.repeat(in_sample_returns[:, :, np.newaxis], 1, axis=2).transpose((1, 0, 2)))  # (T, M, 1)
    B_T_check = np.sum(np.matmul(F_t, F_t.transpose((0, 2, 1))), 0) / (N * T)
    # estimate beta_MSRR
    betas_MSRR = [np.matmul(np.linalg.inv(z_value * np.eye(M) + B_T_check), np.sum(F_t, 0) / (N * T)) for z_value in z]  # M*1
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


def compute_oos_performance_of_MSRR_for_fixed_z(z_value, in_sample_signals,
                                                     in_sample_returns, out_of_sample_signals, out_of_sample_returns):
    """
    This function computes the out-of-sample performance of MSRR portfolios
    :param z_value: a number of ridge penalty parameter, note that it's not a vector
    :param in_sample_signals: in-sample signals with dimension N*M*T
    :param in_sample_returns: in-sample returns with dimension N*T
    :param out_of_sample_signals: out-of-sample signals with dimension N*M*T_test
    :param out_of_sample_returns: out-of-sample returns with dimension N*T_test
    :return: returns and risks (squared returns) of the OLS-ridge portfolios
    """
    beta_MSRR_value = beta_MSRR_function(in_sample_signals, in_sample_returns, z_value)
    loss_return_MSRR, loss_risk_MSRR = oos_performance(beta_MSRR_value, out_of_sample_signals, out_of_sample_returns)
    return loss_return_MSRR, loss_risk_MSRR


def run_new_simulations_for_testing_ols(N_sim, z, N, M, T, T_test, a_Sigma, a_Psi, target_psi):
    """
    this is a new simulations function that
    :param N_sim:
    :param z:
    :param N:
    :param M:
    :param T:
    :param T_test:
    :param a_Sigma:
    :param a_Psi:
    :param target_psi:
    :return:
    """

    PSI_EIGENVALUES = np.load('../plots_of_eigenvalues/eigenvalues_across_signals_1994-01-31T00:00:00.000000000.npy',
                              allow_pickle=True)

    SIGMA_EIGENVALUES = np.load('../plots_of_eigenvalues/eigenvalues_across_stocks_1994-01-31T00:00:00.000000000.npy',
                                allow_pickle=True)

    eig = np.abs(PSI_EIGENVALUES[-M:])

    # sometimes the minimal eigenvalue is already large.
    # we need to correct it to avoid crazy numbers.
    # if eig[0] > 0.001:
    #     eig *= 0.001 / eig[0]
    psi_eigenvalues = np.power(eig, a_Psi)
    Psi = np.diag(psi_eigenvalues)

    # Sigma is the covariance of signals across assets.
    eig = np.abs(SIGMA_EIGENVALUES[-N:])

    # sometimes the minimal eigenvalue is already large.
    # we need to correct it to avoid crazy numbers.
    # if eig[0] > 0.001:
    #     eig *= 0.001 / eig[0]
    sigma_eigenvalues = np.power(eig, a_Sigma)
    Sigma = np.diag(sigma_eigenvalues)

    herf = (np.sum(sigma_eigenvalues) ** 2) / (np.sum(sigma_eigenvalues ** 2))
    sru_math = np.sqrt(herf / (2 + herf * (1 + 1 / target_psi)))
    sruv_math = np.sqrt(herf / (2 + herf / target_psi))

    b = target_psi * M / (np.sum(psi_eigenvalues) * np.sum(sigma_eigenvalues))

    # SRU benchmark
    SRU_benchmark = benchmark_SRU(M, b, Sigma, Psi)

    print(f'comparing two sru: {SRU_benchmark, sru_math, sruv_math}, where herf= {herf}\n '
          f'and targetpsi={target_psi} has led to b={b}')

    # parameters
    c = M / (N * T)
    parameter = (c, N, M, b, T, T_test, N_sim, a_Sigma, a_Psi, sruv_math)

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

    for Sim_i in range(N_sim):
        print(str(Sim_i) + '/' + str(N_sim))
        # DGP for each simulations
        beta, in_sample_signals, out_of_sample_signals, in_sample_returns, out_of_sample_returns \
            = generate_data(N, M, T, b, T_test, Sigma, Psi)

        # A_T (M*M)
        A_T = np.sum(np.matmul(in_sample_signals.transpose((2, 1, 0)),
                               in_sample_signals.transpose((2, 0, 1))), 0) / (N * T)

        # compute the empirical Stieltjes transform for Marcenko-Pastur Theorem check
        t1 = time.time()
        m_hat[:, Sim_i] = [np.trace(np.linalg.inv(z_value * np.eye(M) + A_T) / M) for z_value in z]
        m_Psi_hat[:, Sim_i] = [map_from_m_to_mPsi(m_hat[i, Sim_i], z[i], M / (N * T)) for i in range(z.shape[0])]
        derivative_of_m_hat[:, Sim_i] = [np.trace(np.linalg.matrix_power(z_value * np.eye(M) + A_T, -2) / M) for z_value in z]

        betas_ols = ols_ridge(in_sample_signals, in_sample_returns, z)
        ols_results = oos_performance(betas_ols, out_of_sample_signals, out_of_sample_returns)

        # ER_Pi_MV_oos for Proposition 5 test
        ER_Pi_MV_oos[:, Sim_i] = ols_results[0].flatten() / N
        # ERsq_Pi_MV_oos for Proposition 6 test
        ERsq_Pi_MV_oos[:, Sim_i] = ols_results[1].flatten() / (N ** 2)
        # SRU of MV portfolio
        SRU_MV_oos[:, Sim_i] = ols_results[2].flatten()
        SRUV_MV_oos[:, Sim_i] = ols_results[3].flatten()

        betas_MSRR_hat = betas_MSRR_function(in_sample_signals, in_sample_returns, z)
        msrr_results = oos_performance(betas_MSRR_hat, out_of_sample_signals, out_of_sample_returns)

        ER_Pi_MSRR_oos[:, Sim_i] = msrr_results[0].flatten() / N
        # ERsq_Pi_MV_oos for Proposition 6 test
        ERsq_Pi_MSRR_oos[:, Sim_i] = msrr_results[1].flatten() / (N ** 2)
        # SRU of MV portfolio
        SRU_MSRR_oos[:, Sim_i] = msrr_results[2].flatten()
        SRUV_MSRR_oos[:, Sim_i] = msrr_results[3].flatten()
        t2 = time.time()
        print(f'loop over z took {t2 - t1}')

    return m_hat.mean(1), derivative_of_m_hat.mean(1), m_Psi_hat.mean(1), ER_Pi_MV_oos.mean(1), \
           ERsq_Pi_MV_oos.mean(1), SRU_MV_oos.mean(1), SRUV_MV_oos.mean(1), \
           ER_Pi_MSRR_oos.mean(1), ERsq_Pi_MSRR_oos.mean(1), SRU_MSRR_oos.mean(1), SRUV_MSRR_oos.mean(1), \
           parameter


def run_simulations_for_testing_ols(N_sim, z, N, M, T, b, T_test, k_Sigma, a_Sigma, sigma_squared_Sigma, k_Psi, a_Psi, sigma_squared_Psi):
    """
    This functions run N_sim simulations for a vector of z and get simulation results for Marcenko-Pastur Theorem, Proposition 5, and Proposition 6
    :param N_sim: number of simulations
    :param z: a vector of ridge penalty parameter
    :param N: number of assets
    :param M: number of variables
    :param T: training samples
    :param b: the level of beta, b = ||beta||^2
    :param T_test: the number of test samples
    :return: out-of-sample value of Marcenko-Pastur Theorem, Proposition 5, and Proposition 6
    """

    # Psi is the covariance across signals
    Psi = generate_Sigma_Psi(k_Psi, M, a_Psi, sigma_squared_Psi)
    # Sigma is the covariance of signals across assets.
    Sigma = generate_Sigma_Psi(k_Sigma, N, a_Sigma, sigma_squared_Sigma)
    # SRU benchmark
    SRU_benchmark = benchmark_SRU(M, b, Sigma, Psi)

    # parameters
    c = M / (N * T)
    parameter = (c, N, M, b, T, T_test, N_sim, k_Sigma, a_Sigma, sigma_squared_Sigma, k_Psi, a_Psi, sigma_squared_Psi, SRU_benchmark)

    if SRU_benchmark > 0.1:
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

        # Initialization for MSRR
        ER_Pi_MSRR_oos = np.zeros(shape=(z.shape[0], N_sim))
        ERsq_Pi_MSRR_oos = np.zeros(shape=(z.shape[0], N_sim))
        SRU_MSRR_oos = np.zeros(shape=(z.shape[0], N_sim))

        for Sim_i in range(N_sim):
            print(str(Sim_i)+'/'+str(N_sim))
            # DGP for each simulations
            beta, in_sample_signals, out_of_sample_signals, in_sample_returns, out_of_sample_returns \
                = generate_data(N, M, T, b, T_test, Sigma, Psi)

            # A_T (M*M)
            A_T = np.sum(np.matmul(in_sample_signals.transpose((2, 1, 0)),
                                   in_sample_signals.transpose((2, 0, 1))), 0)/(N*T)

            # compute the empirical Stieltjes transform for Marcenko-Pastur Theorem check
            for i in range(z.shape[0]):
                z_value = z[i]
                m_hat[i, Sim_i] = np.trace(np.linalg.inv(z_value * np.eye(M) + A_T) / M)

                m_Psi_hat[i, Sim_i] = map_from_m_to_mPsi(m_hat[i, Sim_i], z_value, M/(N*T))

                derivative_of_m_hat[i, Sim_i] = np.trace(np.linalg.matrix_power(z_value * np.eye(M) + A_T, -2) / M)

                # OLS
                # loss_return_ols, loss_risk_ols = compute_oos_performance_of_ridge_ols_for_fixed_z(z_value, in_sample_signals,
                #                                                  in_sample_returns, out_of_sample_signals,
                #                                                  out_of_sample_returns)
                beta_ols = ols_ridge(in_sample_signals, in_sample_returns, z_value)
                loss_return_ols, loss_risk_ols, SRU_ols = oos_performance(beta_ols, out_of_sample_signals, out_of_sample_returns)

                # ER_Pi_MV_oos for Proposition 5 test
                ER_Pi_MV_oos[i, Sim_i] = loss_return_ols / N
                # ERsq_Pi_MV_oos for Proposition 6 test
                ERsq_Pi_MV_oos[i, Sim_i] = loss_risk_ols / (N ** 2)
                # SRU of MV portfolio
                SRU_MV_oos[i, Sim_i] = SRU_ols

                # MSRR
                beta_MSRR_hat = beta_MSRR_function(in_sample_signals, in_sample_returns, z_value)
                loss_return_MSRR, loss_risk_MSRR, SRU_MSRR = oos_performance(beta_MSRR_hat, out_of_sample_signals, out_of_sample_returns)
                # print(beta, beta_MSRR_hat)
                ER_Pi_MSRR_oos[i, Sim_i] = loss_return_MSRR / N
                ERsq_Pi_MSRR_oos[i, Sim_i] = loss_risk_MSRR / (N ** 2)
                SRU_MSRR_oos[i, Sim_i] = SRU_MSRR

        return m_hat, derivative_of_m_hat, m_Psi_hat, ER_Pi_MV_oos, ERsq_Pi_MV_oos, SRU_MV_oos, ER_Pi_MSRR_oos, ERsq_Pi_MSRR_oos, SRU_MSRR_oos, parameter

    else:
        para_str = "c%s, N%s, M%s, b%s, T%s, T_test%s, N_sim%s, k_Sigma%s, a_Sigma%s, sigma_squared_Sigma%s, k_Psi%s, a_Psi%s, sigma_squared_Psi%s, SRU_benchmark%.4f" % parameter
        print(para_str)
        return (parameter)
