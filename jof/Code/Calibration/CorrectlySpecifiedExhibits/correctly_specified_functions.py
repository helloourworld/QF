import numpy as np
import multiprocessing as mp
import time
import pandas as pd
from itertools import product
import os
from simulation_functions import run_new_simulations_for_testing_ols
from expectation_functions import *
# from plot_functions import plot_returns_OLSvsMSRR, plot_returnsSQ_OLSvsMSRR, plot_Lemma7_Gamma, plot_MSRR_SRU_Expectation, plot_MSRR_SRUV_Expectation, plot_Marcenko_Pastur_quotient_MV, plot_Marcenko_Pastur_quotient_MSRR
from Psi_Sigma_Sigma_eps_functions import *
from simulation_functions_with_Sigma_epsilon import *
from Eigenvalue_functions_minusz import Stieltjes_transform_Theorem3_psi_eigenvalues, derivative_of_Stieltjes_transform_Theorem3_psi_eigenvalues, m_Psi_z, derivative_of_m_Psi_z
from auxilliary_functions import matrix_square_root
# from testing_msrr_theory_calcs_matmul import check_decomposition_in_lemma_5, check_lemma21
import pickle
from sklearn.metrics import r2_score
plt.rcParams["font.family"] = "Times New Roman"

def norm_of_beta_hat(b_star, c_1, c_M, z_, m_, m_prime):
    '''
    This function computes the 2nd of norm in section 13
    :param b_star:
    :param c_1:
    :param c_M:
    :param z_:
    :param m_:
    :param m_prime:
    :param tr_Psi_2_2:
    :param xi_prime_1_1:
    :param xi_1_1:
    :return:
    '''

    term1 = b_star * c_1/c_M * (1 - 2*z_*m_ + z_**2 * m_prime)
    term2 = c_1 * m_ - z_*c_1*m_prime
    return term1 + term2


if __name__ == '__main__':
    start_time = time.time()

    ##################################################### Parameters ###################################################
    N = 1 # only one stock, fixed
    print('N = %s' % N)
    T = 100  # training periods
    print('T = %s' % T)
    sigma_star = 1

    M_list = np.arange(1, 1001, 1)
    a_Psi = 0
    print('a_Psi = %s' % a_Psi)

    z = np.array([5e-5, 0.01, 0.1, 1, 10, 50]) # shrinkage
    Sigma_eigenvalues_multiplier = 1
    target_psi_star1 = 1 # Vary tr(Psi)
    target_psi = 1

    # Parameter for b generation
    b_decay_power = 1
    b_star0 = 0.2
    b = b_star0 / (N ** b_decay_power)

    # Parameters for Sigma generation
    a_Sigma = 0

    # Parameters for Sigma_eps generation
    a_Sigma_eps = 0

    ################################ Get calibrated values for correctly specified model ###############################

    # Sigma (covariance of signals across assets) generation
    Sigma, sigma_eigenvalues = generate_Sigma_with_empirical_eigenvalues(N, a_Sigma, Sigma_eigenvalues_multiplier)
    # trace_Sigma = np.trace(Sigma)

    # Sigma_eps generation
    Sigma_eps = generate_Sigma_epsilon(N, a_Sigma_eps)

    # Sigma_hat and its eigenvalues
    sigma_hat, sigma_hat_eigenvalues = sigma_hat_function(Sigma, Sigma_eps)

    ER_Pi_MSRR_collect = np.zeros(len(M_list))
    ERsq_Pi_MSRR_collect = np.zeros(len(M_list))
    Expected_SRU_MSRR_collect = np.zeros(len(M_list))

    SRU_MV_collect = np.zeros((len(M_list), len(z)))
    Ret_MV_collect = np.zeros((len(M_list), len(z)))
    Risk_MV_collect = np.zeros((len(M_list), len(z)))
    Rsq_MV_collect = np.zeros((len(M_list), len(z)))
    MSE_MV_collect = np.zeros((len(M_list), len(z)))
    beta_norm_MV_collect = np.zeros((len(M_list), len(z)))

    for M_i in range(len(M_list)):
        # for a_Psi_i in range(len(a_Psi_list)):

        M = M_list[M_i]
        print('M = %s' % M)

        special_str = r'a_Psi %s, a_Sigma %s, a_Sigma_eps %s' % (a_Psi, a_Sigma, a_Sigma_eps) # , alpha_str)
        special_str_title = r'$a_\Psi$ %s, $a_\Sigma$ %s, $a_{\Sigma_{\epsilon}}$ %s' % (a_Psi, a_Sigma, a_Sigma_eps) # , alpha_str)

        # c and c_M
        c = M/(N*T)
        c_M = M/T

        Psi, psi_eigenvalues = generate_Psi_with_empirical_eigenvalues(M, a_Psi, target_psi_star1)

        herf_sigma_hat = np.trace(sigma_hat**2)/(np.trace(sigma_hat)**2)
        print('np.trace(sigma_hat**2)/(np.trace(sigma_hat)**2) = %s' % herf_sigma_hat)

        # Assumption 5
        b_star1 = b * (np.sum(sigma_hat_eigenvalues) ** 2) / N  # 3.865672199579576
        print('b_star1 = %s' % b_star1)
        sigma_star = np.sum(sigma_hat_eigenvalues) / N
        print('sigma_star = %s' % sigma_star)
        psi_star1 = np.trace(Psi) / M  # checked
        print('psi_star1 = %s' % psi_star1)
        psi_star2 = np.trace(np.matmul(Psi, Psi)) / M
        print('psi_star2 = %s' % psi_star2)
        psi_star3 = np.trace(Psi**3) / M
        k_star = psi_star1 * b_star1 + sigma_star

        # true R_squared in Eq. (37)
        Rsq_true = 1 - N / (N + b * psi_star1 * np.trace(sigma_hat))

        # True risk
        SRU_true, returns_true, risks_true = expected_SRU_ols_zto0_N1_True(sigma_hat, b, psi_star1)

        ########### Expectation ###########

        # sigma_hat = I
        # Expected value of Marcenko-Pastur Theorem
        m = marcenko_pastur(1, c, z) # 200
        derivative_of_m = derivative_of_marcenko_pastur(1., m, c, z) # 200
        # Eq. (30) in Prop 5
        xi_z = c_M * m
        derivative_of_xi_z = derivative_of_xi_function(c, z, N, M, m, derivative_of_m, xi_z, sigma_hat)

        # Expected value of Proposition 7 in the paper
        nu_z = nu_function(xi_z, z, c, Psi, M)
        ER_Pi_MV = expected_return_ols(nu_z, sigma_hat, b)

        # Expected value of Proposition 9 in the paper
        derivative_of_nu_z = derivative_of_nu_function(c, z, xi_z, derivative_of_xi_z)
        ERsq_Pi_MV = expected_squared_return_ols(c, z, b, M, Psi, sigma_hat, nu_z, derivative_of_nu_z)

        Expected_SRU_MV = ER_Pi_MV/np.sqrt(ERsq_Pi_MV)

        Expected_SRU_MV_check = expected_SRU_ols_z_NFinite(sigma_hat, nu_z, z, b, c, psi_star1, derivative_of_nu_z)
        SRU_MV_zto0 = expected_SRU_ols_zto0_NFinite(sigma_hat, b, c, psi_star1)

        Expected_SRUV_MV = np.sqrt(Expected_SRU_MV ** 2 / (1 - Expected_SRU_MV ** 2))

        # Eq. (35)
        SST = N + b*psi_star1*np.trace(sigma_hat)
        # Eq. (34)
        MSE = SST - np.trace(sigma_hat)*b*psi_star1 + xi_z + derivative_of_xi_z*(z - z**2/c*np.trace(sigma_hat)*b)

        # beta norm
        beta_norm = norm_of_beta_hat(b, c_M, c_M, z, m, derivative_of_m)

        SRU_MV_collect[M_i, :] = Expected_SRU_MV
        Ret_MV_collect[M_i, :] = ER_Pi_MV
        Risk_MV_collect[M_i, :] = ERsq_Pi_MV
        Rsq_MV_collect[M_i, :] = 1-MSE/SST
        MSE_MV_collect[M_i, :] = MSE
        beta_norm_MV_collect[M_i, :] = beta_norm

    ###################################################### Save data ###################################################
    data_save_path = './CorrectSpec_data/'
    if not os.path.exists(data_save_path):
        os.mkdir(data_save_path)

    pd.DataFrame(Rsq_MV_collect).to_csv(data_save_path + "R2.csv", header=None, index=None)
    pd.DataFrame(Ret_MV_collect).to_csv(data_save_path + "ER.csv", header=None, index=None)
    pd.DataFrame(Risk_MV_collect).to_csv(data_save_path + "Vol.csv", header=None, index=None)
    pd.DataFrame(SRU_MV_collect).to_csv(data_save_path + "SR.csv", header=None, index=None)
    pd.DataFrame(MSE_MV_collect).to_csv(data_save_path + "MSE.csv", header=None, index=None)
    pd.DataFrame(beta_norm_MV_collect).to_csv(data_save_path + "Bnorm.csv", header=None, index=None)
    pd.DataFrame([Rsq_true, returns_true, risks_true, SRU_true]).to_csv(data_save_path + "TRUE_Rsq_ET_Vol_SR.csv", header=None, index=None)

    # Changed to Matlab for plotting


