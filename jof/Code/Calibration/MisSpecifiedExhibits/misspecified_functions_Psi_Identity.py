import numpy as np
import pandas as pd
import time
import os
from main import xi_as_a_function_of_m, xi_prime_as_a_function_of_m
from expectation_functions import marcenko_pastur, derivative_of_marcenko_pastur, derivative_of_nu_function, expected_SRU_ols_zto0_N1_True
from plot_functions import series_plot_for_misspecified, series_plot_for_misspecified_final

def xi_hat(xi_1_1, xi_prime_1_1, c_1, m, m_prime, list_of_z):
    '''
    This function computes \hat{xi}_{1,1}(z) in Eq. (138)
    :param xi_1_1:
    :param xi_prime_1_1:
    :param c_1:
    :param m:
    :param m_prime:
    :return:
    '''
    numerator = (xi_1_1 + list_of_z * xi_prime_1_1)/c_1 * xi_prime_1_1
    denominator = -m + list_of_z * m_prime
    return numerator/denominator

def Delta_z_Psi22_is_Identity(M_2, M, b_star, xi_hat_1_1, xi_1_1):
    '''
    This function computes \Delta(z) in Eq. (142) with Psi_{2,2} = I
    :param Psi_2_2:
    :param b_star:
    :param xi_hat_1_1:
    :param M:
    :return:
    '''
    Delta = M_2/M * b_star * xi_hat_1_1 * (1 + xi_1_1)**(-2)
    return Delta

def trace_Psi11_E_betahat_betahatprime(b_star, c_1, c_M, psi_star1, list_of_z, xi_z, xi_prime_z, Delta):
    '''
    This function computes the value of tr(Psi_{1,1} E[\hat{beta}\hat{beta}']) in Eq. (140)
    :param b_star:
    :param c_1:
    :param c_M:
    :param psi_star1:
    :param list_of_z:
    :param xi_z:
    :param xi_prime_z:
    :param Delta:
    :return:
    '''
    term1 = b_star*c_1/c_M * (psi_star1 - 2*list_of_z/c_1*xi_z - list_of_z**2/c_1*xi_prime_z)
    return term1 + xi_z + list_of_z*xi_prime_z + Delta

def trace_Psi11_E_betahat_betahatprime_as_function_of_nu(b_star, c_1, c_M, nu_z_c1, nu_prime_z_c1, list_of_z, Delta):
    '''
    This function computes the value of tr(Psi_{1,1} E[\hat{beta}\hat{beta}']) in Eq. (208), which is a function of \nu
    :param b_star:
    :param c_1:
    :param c_M:
    :param psi_star1:
    :param list_of_z:
    :param xi_z:
    :param xi_prime_z:
    :param Delta:
    :return:
    '''
    out = b_star * c_1 / c_M * (nu_z_c1 + list_of_z * nu_prime_z_c1) - c_1* nu_prime_z_c1 + Delta
    return out


def nu_function_PsiIdentity(xi_1_1, z_, c_1, psi_star1 = 1):
    '''
    This function computes \nu(z) according to Eq. (207) with \Psi_{1,1} = I
    :param xi:
    :param z:
    :param c:
    :param Psi:
    :param M:
    :return:
    '''
    nu = psi_star1 - 1/c_1 * z_ * xi_1_1
    return nu

def MSE_OLS_with_M_Infty(b_star, M, M_1, beta_Psi11_Ebetahat, tr_Psi11_E_betahat_betahatprime):
    '''
    This function returns MSE_OLS in Eq. (204)
    :param b_star:
    :param M:
    :param M_1:
    :param beta_Psi11_Ebetahat: from Eq. (208)
    :param tr_Psi11_E_betahat_betahatprime: from Eq. (208)
    :return:
    '''
    MSE = 1 + b_star - 2*beta_Psi11_Ebetahat + tr_Psi11_E_betahat_betahatprime
    return MSE

def SRU_OLS_with_M_Infty(b_star, M, M_1, M_2, beta_Psi11_Ebetahat, tr_Psi11_E_betahat_betahatprime):
    '''
    This function returns SRU_OLS in Eq. (204)
    :param b_star:
    :param M:
    :param M_1:
    :param M_2:
    :param beta_Psi11_Ebetahat:
    :param tr_Psi11_E_betahat_betahatprime:
    :param beta_Psi_Ebetahat:
    :return:
    '''

    returns = beta_Psi11_Ebetahat

    # denominator
    term1 = 2*beta_Psi11_Ebetahat**2
    term2 = (1 + b_star/M*(M_1 + M_2)) * tr_Psi11_E_betahat_betahatprime
    risks = term1 + term2
    SRU = returns/np.sqrt(risks)
    return SRU, returns, risks

def norm_of_beta_hat(b_star, c_1, c_M, z_, m_, m_prime, tr_Psi_2_2, xi_prime_1_1, xi_1_1):
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
    term3 = tr_Psi_2_2/M * b_star * xi_prime_1_1 * (1 + xi_1_1)** (-2)
    return term1 + term2 + term3

def theoretical_functions_for_misspecified_model(M, M1_list, T, list_of_z, b_star):
    '''
    This function returns the theoretical values of the mis-specified model in Proposition 5
    :param xi_1_1:
    :param xi_prime_1_1:
    :param c_1:
    :param c_M:
    :param m:
    :param m_prime:
    :param list_of_z:
    :param Psi_2_2:
    :param b_star:
    :param psi_star1:
    :param xi_z:
    :param xi_prime_z:
    :return:
    '''

    M2_list = M - M1_list
    c_1 = M1_list / T
    c_M = M/T

    # m, m_prime
    m_c1 = np.array([marcenko_pastur(1, c_1, z_) for z_ in list_of_z]) # len(list_of_z) \times len(c_1)
    m_prime_c1 = np.array([derivative_of_marcenko_pastur(1., m_c1[z_idx, :], c_1, list_of_z[z_idx]) for z_idx in range(len(list_of_z))]) # len(list_of_z) \times len(c_1)

    # xi_1_1, xi_prime_1_1
    xi_1_1 = np.array([xi_as_a_function_of_m(m_c1[:, idx], c_1[idx], list_of_z).flatten() for idx in range(len(c_1))]).T# len(list_of_z) \times len(c_1)
    xi_prime_1_1 = np.array([xi_prime_as_a_function_of_m(m_prime_c1[:, idx], m_c1[:, idx], xi_1_1[:, idx], c_1[idx], list_of_z).flatten() for idx in range(len(c_1))]).T # len(list_of_z) \times len(c_1)

    # xi_hat_1_1
    xi_hat_1_1 = np.array([xi_hat(xi_1_1[:, idx], xi_prime_1_1[:, idx], c_1[idx], m_c1[:,idx], m_prime_c1[:,idx], list_of_z) for idx in range(len(c_1))]).T # len(list_of_z) \times len(c_1)

    # Delta(z)
    Delta = np.array([Delta_z_Psi22_is_Identity(M2_list, M, b_star, xi_hat_1_1[z_idx, :], xi_1_1[z_idx, :]) for z_idx in range(len(list_of_z))]) # len(list_of_z) \times len(c_1)

    # nu(z, c_1) and nu'(z, c_1)
    nu_z_c1 = np.array([nu_function_PsiIdentity(xi_1_1[z_idx, :], list_of_z[z_idx], c_1) for z_idx in range(len(list_of_z))]) # len(list_of_z) \times len(c_1)
    nu_prime_z_c1 = np.array([derivative_of_nu_function(c_1, list_of_z[z_idx], xi_1_1[z_idx, :], xi_prime_1_1[z_idx, :]) for z_idx in range(len(list_of_z))]) # len(list_of_z) \times len(c_1)

    # \beta'Psi_{1,1}E[\hat{\beta}]
    beta_Psi11_Ebetahat = b_star * c_1/c_M * nu_z_c1 # len(list_of_z) \times len(c_1)

    # z = 0
    m_star_c1_z0 = m_c1[0,:] - (1-1/c_1)/list_of_z[0]
    nu_z_c1_z0 = psi_star1 - 1/(c_1**2)/m_star_c1_z0
    beta_Psi11_Ebetahat_z0 = b_star * c_1 / c_M * nu_z_c1_z0 # len(list_of_z) \times len(c_1)
    beta_Psi11_Ebetahat[0,:] = beta_Psi11_Ebetahat_z0

    tr_Psi11_E_betahat_betahatprime = np.array([trace_Psi11_E_betahat_betahatprime_as_function_of_nu(b_star, c_1, c_M, nu_z_c1[z_idx, :], \
                                      nu_prime_z_c1[z_idx, :], list_of_z[z_idx], Delta[z_idx, :]) for z_idx in range(len(list_of_z))]) # len(list_of_z) \times len(c_1)

    MSE_OLS = np.array([MSE_OLS_with_M_Infty(b_star, M, M1_list, beta_Psi11_Ebetahat[z_idx, :], \
                                             tr_Psi11_E_betahat_betahatprime[z_idx, :]) for z_idx in range(len(list_of_z))])# len(list_of_z) \times len(c_1)

    SRU_OLS = np.array([SRU_OLS_with_M_Infty(b_star, M, M1_list, M2_list, beta_Psi11_Ebetahat[z_idx, :], \
              tr_Psi11_E_betahat_betahatprime[z_idx, :])[0] for z_idx in range(len(list_of_z))]) # len(list_of_z) \times len(c_1)
    Returns_OLS = np.array([SRU_OLS_with_M_Infty(b_star, M, M1_list, M2_list, beta_Psi11_Ebetahat[z_idx, :], \
                                             tr_Psi11_E_betahat_betahatprime[z_idx, :])[1] for z_idx in range(len(list_of_z))])
    Risks_OLS = np.array([SRU_OLS_with_M_Infty(b_star, M, M1_list, M2_list, beta_Psi11_Ebetahat[z_idx, :], \
                                             tr_Psi11_E_betahat_betahatprime[z_idx, :])[2] for z_idx in range(len(list_of_z))])
    Rsq_OLS = 1 - MSE_OLS/(1 + b_star)

    # true R_squared
    Rsq_true = 1 - N / (N + b_star * psi_star1 * N)

    # True risk
    SRU_true, returns_true, risks_true = expected_SRU_ols_zto0_N1_True(np.eye(N), b_star, psi_star1)

    # b-norm
    beta_norm = np.array([norm_of_beta_hat(b_star, c_1, c_M, list_of_z[z_idx], m_c1[z_idx, :], m_prime_c1[z_idx, :], \
                                           M - M1_list, xi_prime_1_1[z_idx, :], xi_1_1[z_idx, :]) for z_idx in
                          range(len(list_of_z))])

    return MSE_OLS, SRU_OLS, Returns_OLS, Risks_OLS, Rsq_OLS, Rsq_true, SRU_true, returns_true, risks_true, beta_norm

def parameter_strings_for_misspecified():
    para_str = ' M %s, T %s' % (M, T)
    para_str_title = ' M %s, T %s' % (M, T)
    return para_str, para_str_title

if __name__ == '__main__':
    start_time = time.time()

    ##################################################### Parameters ###################################################
    list_of_z = np.array([5e-5, 0.01, 0.1, 1, 10, 50])
    N = 1
    M = 1000
    T = 100
    M1_list = np.arange(1, M + 1)

    # Parameter for b generation
    b_decay_power = 1
    b_star0 = 0.2
    b_star = b_star0 / (N ** b_decay_power)
    psi_star1 = 1

    Psi = np.eye(M)
    performance_names = ['R-square', 'SRU', 'MSE']

    ################################### Get calibrated values for mis-specified model ##################################
    MSE_OLS, SRU_OLS, Returns_OLS, Risks_OLS, Rsq_OLS, Rsq_true, SRU_true, returns_true, risks_true, beta_norm = \
        theoretical_functions_for_misspecified_model(M, M1_list, T, list_of_z, b_star)

    ###################################################### Save data ###################################################
    data_save_path = './MisSpec_data/'
    if not os.path.exists(data_save_path):
        os.mkdir(data_save_path)
    para_str, para_str_title = parameter_strings_for_misspecified()

    pd.DataFrame(Rsq_OLS).to_csv(data_save_path + "R2.csv", header=None, index=None)
    pd.DataFrame(Returns_OLS).to_csv(data_save_path + "ER.csv", header=None, index=None)
    pd.DataFrame(Risks_OLS).to_csv(data_save_path + "Vol.csv", header=None, index=None)
    pd.DataFrame(SRU_OLS).to_csv(data_save_path + "SR.csv", header=None, index=None)
    pd.DataFrame(MSE_OLS).to_csv(data_save_path + "MSE.csv", header=None, index=None)
    pd.DataFrame(beta_norm).to_csv(data_save_path + "Bnorm.csv", header=None, index=None)
    pd.DataFrame([Rsq_true, returns_true, risks_true, SRU_true]).to_csv(data_save_path + "TRUE_Rsq_ET_Vol_SR.csv",
                                                                        header=None, index=None)










