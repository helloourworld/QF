# This file contains functions for computing expectations.
import numpy as np
from auxilliary_functions import matrix_square_root
# import scipy

def derivative_kappa_z(z, c_M, q_minus_z, derivative_of_q_minus_z):
    '''
    This function computes the first derivative of kappa(z)
    :param z:
    :param c_M:
    :param q_minus_z:
    :param derivative_of_q_minus_z:
    :return:
    '''
    kappa_z_theory_1st_derivative_numerator = 1 / c_M * (-q_minus_z + z * derivative_of_q_minus_z)
    kappa_z_theory_1st_derivative_denominator = (1 / c_M - 1 + z * q_minus_z) ** 2
    kappa_z_theory_1st_derivative = kappa_z_theory_1st_derivative_numerator / kappa_z_theory_1st_derivative_denominator
    return kappa_z_theory_1st_derivative

def Gamma1_z_Lemma6(b, z, sigma_star, b_star1, k_star, kappa_z_theory, M, Psi):
    '''
    This function computes the Gamma_1(z) by Lemma 6
    :param b:
    :param z:
    :param sigma_star:
    :param b_star1:
    :param k_star:
    :param kappa_z_theory:
    :param M:
    :param Psi:
    :return:
    '''
    p_z = -1 / sigma_star * z * (1 + kappa_z_theory)
    m_Psi_p_z = [1 / M * np.sum((np.diag(Psi) - p_z[z_idx]) ** (-1)) for z_idx in range(len(z))]
    hat_Gamma_1_numerator = b / z * (1 + p_z * m_Psi_p_z)
    hat_Gamma_1_denominator = b_star1 * (p_z * m_Psi_p_z + 1) + k_star / p_z
    hat_Gamma_1 = -sigma_star * hat_Gamma_1_numerator / hat_Gamma_1_denominator
    return hat_Gamma_1

def derivative_of_Gamma1_z(b, z, sigma_star, b_star1, k_star, kappa_z_theory, M, Psi, kappa_z_theory_1st_derivative):
    '''
    This function computes the first derivative of Gamma_1(z) by Lemma 6
    :param b:
    :param z:
    :param sigma_star:
    :param b_star1:
    :param k_star:
    :param kappa_z_theory:
    :param M:
    :param Psi:
    :return:
    '''
    p_z = -1 / sigma_star * z * (1 + kappa_z_theory)
    derivative_p_z = -1 / sigma_star * (1 + kappa_z_theory + z * kappa_z_theory_1st_derivative)

    m_Psi_p_z = np.array([1 / M * np.sum((np.diag(Psi) - p_z[z_idx]) ** (-1)) for z_idx in range(len(z))])
    derivative_of_m_Psi_p_z = np.array([1 / M * np.sum((np.diag(Psi) - p_z[z_idx]) ** (-2)) for z_idx in range(len(z))])

    derivative_factor1 = ((-1/z ** (2)) * (1 + p_z * m_Psi_p_z) + 1 / z * (
                derivative_p_z * m_Psi_p_z + p_z * derivative_of_m_Psi_p_z * derivative_p_z)) \
                         / (b_star1 * (p_z * m_Psi_p_z + 1) + k_star / p_z)
    derivative_factor2 = 1 / z * (1 + p_z * m_Psi_p_z) * (-(b_star1 * (p_z * m_Psi_p_z + 1) + k_star / p_z) ** (-2)) * (
                b_star1 * (derivative_p_z * m_Psi_p_z + p_z * derivative_of_m_Psi_p_z * derivative_p_z) \
                - k_star * p_z ** (-2) * derivative_p_z)
    derivative_of_Gamma_1 = -sigma_star * b * (derivative_factor1 + derivative_factor2)
    return derivative_of_Gamma_1

def derivative_of_Gamma2_z(b, z, psi_star1, k_star, kappa_z_theory, kappa_z_theory_1st_derivative, Gamma1_z, Gamma1_z_prime):
    '''
    This function computes the first derivative of Gamma_2(z) according to Lemma 6
    :param b:
    :param z:
    :param psi_star1:
    :param k_star:
    :param kappa_z_theory:
    :param kappa_z_theory_1st_derivative:
    :param Gamma1_z:
    :param Gamma1_z_prime:
    :return:
    '''
    factor1 = -(Gamma1_z + z*Gamma1_z_prime)*(1+kappa_z_theory)
    factor2 = (b*psi_star1 - z*Gamma1_z)*(kappa_z_theory_1st_derivative)
    derivative_of_Gamma2_z = 1/k_star * (factor1 + factor2)
    return derivative_of_Gamma2_z


def Gamma1_hat_Lemma29(b, q, K, z, mu_Kminus1_q, mu_K_q, epsilon_0, epsilon_1, sigma_star, b_star1, psi_star1, k_star, kappa_z_theory):
    A = -b * q ** (-K) / z * (mu_Kminus1_q - 1) - epsilon_0 / z * b
    B = 1 / sigma_star * b_star1 * q ** (-K) * (-1 + mu_K_q - psi_star1 * q) - q ** (
                -K + 1) - epsilon_0 / z * k_star / (1 + kappa_z_theory) + epsilon_1
    Gamma1_T_hat = A / B
    return Gamma1_T_hat

def marcenko_pastur(sigma_squared, c_, z_): # No Sigma_epsilon
    """
    This function computes the value of Marcenko-Pastur Theorem: m(-z, c)
    please ignore sigma_squared for now
    :param sigma_squared: set as 1
    :param c: c = M/(N*T)
    :param z: a vector of ridge penalty parameter
    :return: value of m(-z)
    """
    sqrt_term = np.sqrt((sigma_squared * (1 - c_) + z_) ** 2 + 4 * c_ * sigma_squared * z_)
    tmp = - (sigma_squared * (1 - c_) + z_) + sqrt_term
    tmp = tmp / (2 * c_ * sigma_squared * z_)
    return tmp
    # sqrt_term = np.sqrt((sigma_squared * (1 - c_) + z_) ** 2 + 4 * c_ * sigma_squared * z_)
    # numerator = sqrt_term + (sigma_squared * (1 - c_) + z_)
    # return 2/numerator


def derivative_of_marcenko_pastur(sigma_squared, m, c, z): # No Sigma_epsilon
    """
    This function computes the derivative of Marcenko-Pastur Theorem: m'(-z, c)
    please ignore sigma_squared for now
    :param sigma_squared: set as 1
    :param m: m = marcenko_pastur(1, c, z)
    :param c: c = M/(N*T)
    :param z: a vector of ridge penalty parameter
    :return: the value of m'(-z, c)
    """
    numerator = c * sigma_squared * (m ** 2) + m
    denominator = 2 * c * sigma_squared * z * m + (sigma_squared * (1 - c) + z)
    tmp = numerator / denominator
    return tmp


def xi_function(c, z, m): # Only works when Sigma is an identity matrix
    """
    This function computes xi(z) by Eq. (18) in Prop 15
    :param c: c = M/(N*T)
    :param z: a vector of ridge penalty parameter
    :param m: m = marcenko_pasturxi_function(1, c, z)
    :return: xi = (1 - z * m) / ((1 / c) - 1 + z * m)
    """
    return (1 - z * m) / ((1 / c) - 1 + z * m)


def derivative_of_xi_function(c, z, N, M, m_minus_z, derivative_of_m_minus_z, xi_z, sigma_hat):
    """
    This function computes xi'(z) by Lemma 16
    :param c: c = M/(N*T)
    :param z: z = a vector of ridge penalty parameter
    :param m: m = marcenko_pastur(1, c, z)
    :param derivative_of_m:
    :param xi: xi = xi_function(c, z, m)
    :return: xi'(z)
    """
    # return c * (z * derivative_of_m - m) * (1 + xi) ** 2
    derivative_of_xi_z_numerator = z*derivative_of_m_minus_z - m_minus_z
    # derivative_of_xi_z_numerator_hat = z*derivative_of_m_hat - m_hat
    inv_I_xi_Sigmahat = [np.linalg.inv(np.eye(N) + xi_z[z_idx]*sigma_hat) for z_idx in range(len(z))]
    derivative_of_xi_z_denominator = np.array([1/(c*N) * np.trace(np.matmul(sigma_hat, \
                                np.matmul(inv_I_xi_Sigmahat[z_idx], inv_I_xi_Sigmahat[z_idx]))) for z_idx in range(len(z))])
    return derivative_of_xi_z_numerator/derivative_of_xi_z_denominator

def nu_function(xi, z, c, Psi, M):
    '''
    This function computes \nu(z) according to Eq. (32) in Prop 7
    :param xi:
    :param z:
    :param c:
    :param Psi:
    :param M:
    :return:
    '''
    nu = np.trace(Psi)/M - z*xi/c
    return nu

def derivative_of_nu_function(c, z, xi, xi_prime):
    '''
    This function computes the first derivative of \nu(z) according to Eq. (38)
    :param c:
    :param z:
    :param xi:
    :param xi_prime:
    :return:
    '''
    nu_prime = - (1 / c) * (xi + z * xi_prime)
    return nu_prime


def nu_function_Ninfinity(M, sigma_star, psi_eigenvalues, z):
    '''
    This function computes \nu(z) by Eq. (347), with the condition N to infinity
    :param M:
    :param sigma_star:
    :param psi_eigenvalues:
    :param z:
    :return:
    '''
    nu = np.array([1/M*np.sum(sigma_star * psi_eigenvalues**2/(sigma_star * psi_eigenvalues + z_)) for z_ in z])
    return nu

def derivative_of_nu_Ninfinity(M, sigma_star, psi_eigenvalues, z):
    '''
    This function computes the first derivative of \nu(z) according to Eq. (38)
    :param M:
    :param sigma_star:
    :param psi_eigenvalues:
    :param z:
    :return:
    '''
    nu_prime = - np.array([1/M*np.sum(sigma_star * psi_eigenvalues**2/((sigma_star * psi_eigenvalues + z_)**2)) for z_ in z])
    return nu_prime


def expected_return_ols(nu_z, Sigma_hat, b):
    """
    This function computes the expected returns of OLS portfolios
    :param norm_beta_squared: the level of beta, b = ||beta||^2
    :param c: c = M/(N*T)
    :param z: a vector of ridge penalty parameter
    :param xi: xi = xi_function(c, z, m)
    :return: norm_beta_squared * (1 - z * xi / c)
    """
    MV_ret = np.trace(Sigma_hat) * b * nu_z
    return MV_ret


# def expected_return_ols(norm_beta_squared, c, z, xi):
#     """
#     This function computes the expected returns of OLS portfolios
#     :param norm_beta_squared: the level of beta, b = ||beta||^2
#     :param c: c = M/(N*T)
#     :param z: a vector of ridge penalty parameter
#     :param xi: xi = xi_function(c, z, m)
#     :return: norm_beta_squared * (1 - z * xi / c)
#     """
#     nu = 1 - z * xi / c
#     return norm_beta_squared * nu

def expected_squared_return_ols(c, z, b, M, Psi, sigma_hat, nu, nu_prime):
    """
    Proposition 9 in the paper
    :param N: number of assets
    :param norm_beta_squared: the level of beta, b = ||beta||^2
    :param c: c = M/(N*T)
    :param z: a vector of ridge penalty parameter
    :param xi: xi = xi_function(c, z, m)
    :param xi_prime: xi_prime = derivative_of_xi_function(c, z, m, derivative_of_m, xi)
    :return: value of Proposition 6
    """
    term1 = b**2 * ((np.trace(sigma_hat))**2 + np.trace(sigma_hat**2)) * nu**2
    factor = np.trace(sigma_hat) + b/M* np.trace(Psi) * np.trace(sigma_hat**2)
    term2 = factor * (b*nu + (b*z - c) * nu_prime)
    return term1 + term2

def expected_SRU_ols_z_NInfinity(sigma_star, b_star0, nu_z, z, c_M, nu_prime_z):
    '''
    This function computes SRU in Eq. (30) with N to infinity
    :param sigma_star:
    :param b_star0:
    :param nu_z:
    :param z:
    :param c_M:
    :param nu_prime_z:
    :return:
    '''
    numerator = sigma_star * b_star0 * nu_z
    denominator = np.sqrt(numerator**2 + sigma_star * (b_star0 * nu_z + (b_star0 * z - c_M)*nu_prime_z))
    return numerator/denominator


def expected_SRU_ols_zto0_NInfinity(sigma_star, b_star0, psi_star1, c_M):
    """
    Proposition 9 in the paper with N to infinity
    :param N: number of assets
    :param norm_beta_squared: the level of beta, b = ||beta||^2
    :param c: c = M/(N*T)
    :param z: a vector of ridge penalty parameter
    :param xi: xi = xi_function(c, z, m)
    :param xi_prime: xi_prime = derivative_of_xi_function(c, z, m, derivative_of_m, xi)
    :return: value of Proposition 6
    """
    numerator = sigma_star * b_star0 * psi_star1
    denominator = np.sqrt(numerator**2 + sigma_star * (b_star0 * psi_star1 + c_M/sigma_star))
    return numerator/denominator

# Functions with N in Finite, like N = 1
def expected_SRU_ols_z_NFinite(sigma_hat, nu_z, z, b, c, psi_star1, derivative_of_nu_z):
    '''
    This function computes SRU in Eq. (63) with finite N
    :param sigma_star:
    :param b_star0:
    :param nu_z:
    :param z:
    :param c_M:
    :param nu_prime_z:
    :return:
    '''
    numerator = np.trace(sigma_hat)*b * nu_z
    denominator = np.sqrt(b**2*(np.trace(sigma_hat)**2 + np.trace(sigma_hat**2)) * nu_z**2 +\
                          (np.trace(sigma_hat) + b*psi_star1*np.trace(sigma_hat**2))*(b*nu_z + (b*z - c)*derivative_of_nu_z) )
    return numerator/denominator


def expected_SRU_ols_zto0_NFinite(sigma_hat, b, c, psi_star1):
    """
    This function computes SRU in Eq. (63) with finite N
    :param N: number of assets
    :param norm_beta_squared: the level of beta, b = ||beta||^2
    :param c: c = M/(N*T)
    :param z: a vector of ridge penalty parameter
    :param xi: xi = xi_function(c, z, m)
    :param xi_prime: xi_prime = derivative_of_xi_function(c, z, m, derivative_of_m, xi)
    :return: value of Proposition 6
    """
    numerator = np.trace(sigma_hat) * b * psi_star1
    denominator = np.sqrt(b**2*(np.trace(sigma_hat)**2 + np.trace(sigma_hat**2)) * psi_star1**2 +\
                          (np.trace(sigma_hat) + b*psi_star1*np.trace(sigma_hat**2))*(b*psi_star1 + 1/(1/c - 1)) )
    return numerator/denominator

def expected_SRU_ols_zto0_N1_True(sigma_hat, b, psi_star1):
    """
    This function computes SRU in Eq. (63) with finite N
    :param N: number of assets
    :param norm_beta_squared: the level of beta, b = ||beta||^2
    :param c: c = M/(N*T)
    :param z: a vector of ridge penalty parameter
    :param xi: xi = xi_function(c, z, m)
    :param xi_prime: xi_prime = derivative_of_xi_function(c, z, m, derivative_of_m, xi)
    :return: value of Proposition 6
    """
    returns = np.trace(sigma_hat) * b * psi_star1
    risks = b**2*(np.trace(sigma_hat)**2 + np.trace(sigma_hat**2)) * psi_star1**2 +\
                          (np.trace(sigma_hat) + b*psi_star1*np.trace(sigma_hat**2))*(b*psi_star1 + 0)
    SRU = returns/np.sqrt(risks)
    return SRU, returns, risks


# def expected_squared_return_ols(N, norm_beta_squared, c, z, xi, xi_prime):
#     """
#     Proposition 6 in the paper
#     :param N: number of assets
#     :param norm_beta_squared: the level of beta, b = ||beta||^2
#     :param c: c = M/(N*T)
#     :param z: a vector of ridge penalty parameter
#     :param xi: xi = xi_function(c, z, m)
#     :param xi_prime: xi_prime = derivative_of_xi_function(c, z, m, derivative_of_m, xi)
#     :return: value of Proposition 6
#     """
#     nu = (1 - z * xi / c)
#     nu_prime = - (1 / c) * (xi + z*xi_prime)
#     term1 = (N ** 2 + N) * norm_beta_squared ** 2 * (nu ** 2)
#     term2 = (N + N * norm_beta_squared) * (norm_beta_squared * nu + (norm_beta_squared * z - c) * nu_prime)
#     return (term1 + term2)/(N ** 2)

# def expected_return_MSRR(z, b_star1, sigma_star, psi_star1, c_M, kappa_z):
#     """
#     This function computes the expected return of MSRR
#     :param z:
#     :param b_star0:
#     :param b_star1:
#     :param sigma_star:
#     :param psi_star1:
#     :param c_M:
#     :param kappa_z:
#     :return:
#     """
#     # term1 = b_star0 * sigma_star**2
#     term2 = psi_star1 - z/(sigma_star* c_M)*kappa_z
#     term3 = psi_star1*b_star1 + sigma_star
#     MSRR_expected_return = b_star1 * term2/term3
#     return MSRR_expected_return





def expected_return_MSRR(sigma_star, N, Gamma2_z, kappa_z):
    """
    # This function computes the expected return of MSRR by Proposition 13
    :param z:
    :param sigma_star:
    :param N:
    :param Gamma2_z:
    :param kappa_z:
    :return:
    """

    MSRR_expected_return = sigma_star**2 * N * Gamma2_z/(1+kappa_z)
    return MSRR_expected_return


def expected_squared_return_MSRR(z, b, b_star0, b_star1, sigma_star, psi_star1, k_star, c_M, kappa_z_theory, M, Psi, q_minus_z, derivative_of_q_minus_z, N):
    '''
    Proposition 14 in the paper about expected squared returns of MSRR
    :param z:
    :param b:
    :param b_star1:
    :param b_star2:
    :param sigma_star:
    :param psi_star1:
    :param k_star:
    :param c_M:
    :param kappa_z_theory:
    :param M:
    :param Psi:
    :return:
    '''

    # Lemma 8
    kappa_z_theory_prime = derivative_kappa_z(z, c_M, q_minus_z, derivative_of_q_minus_z)

    Gamma1_z = Gamma1_z_Lemma6(b, z, sigma_star, b_star1, k_star, kappa_z_theory, M, Psi)
    Gamma0_z = 1/z*(b - Gamma1_z*k_star/(1+kappa_z_theory))  # b / (z + k_star / (1 + kappa_z_theory))
    Gamma2_z = (b*psi_star1 - z*Gamma1_z)/k_star*(1+kappa_z_theory) # b * (1 + kappa_z_theory) * (psi_star1 - z / (sigma_star * c_M) * kappa_z_theory) / k_star

    Gamma1_z_prime = derivative_of_Gamma1_z(b, z, sigma_star, b_star1, k_star, kappa_z_theory, M, Psi, kappa_z_theory_prime)
    Gamma2_z_prime = derivative_of_Gamma2_z(b, z, psi_star1, k_star, kappa_z_theory, kappa_z_theory_prime, Gamma1_z, Gamma1_z_prime)

    # Gamma3_z_numerator0 = 1 - (-z ** 2 * derivative_of_q_minus_z + 2 * z * q_minus_z + 1 / c_M * (
    #             (kappa_z_theory / (1 + kappa_z_theory)) ** 2))
    Gamma3_z_numerator = 1 - (-z ** 2 * derivative_of_q_minus_z + 2 * z * q_minus_z + c_M * ((1-z*q_minus_z)**2))
    Gamma3_z_denominator = sigma_star ** 2 / ((1 + kappa_z_theory) ** 4)
    Gamma3_z = Gamma3_z_numerator / Gamma3_z_denominator

    # Eq. (40)
    Gamma4_z_numerator = Gamma2_z + z * Gamma2_z_prime - b * b_star1 * (Gamma2_z / b) ** 2 * (
                1 + kappa_z_theory) ** (-2)
    Gamma4_z_denominator = sigma_star * (1 + kappa_z_theory) ** (-2)
    Gamma4_z = Gamma4_z_numerator / Gamma4_z_denominator
    # Gamma4_z = -Gamma2_z_1st_derivative

    # Prop 12, Eq. (38)
    ERsq_Pi_MSRR_factor1 = sigma_star ** 2 * ((1 + kappa_z_theory) ** (-2)) * c_M * Gamma3_z
    ERsq_Pi_MSRR_factor2 = ((1 + kappa_z_theory) ** (-2)) * (
        (b_star1 ** 2) * (Gamma2_z ** 2) / (b ** 2) + b_star1 * sigma_star * Gamma4_z / b)
    # ERsq_Pi_MSRR_factor3 = 2 * (Gamma2_z / b * b_star1) * c_M * ((sigma_star + psi_star1 * b_star2) ** 2) * Gamma3_z / (
    #             (1 + kappa_z_theory) ** 3)
    ERsq_Pi_MSRR_factor3 = 2 * (Gamma2_z / b * b_star1) * c_M * (sigma_star ** 2) * Gamma3_z / ((1 + kappa_z_theory) ** 3)
    ERsq_Pi_MSRR = ERsq_Pi_MSRR_factor1 + ERsq_Pi_MSRR_factor2 - ERsq_Pi_MSRR_factor3

    Q = Gamma3_z_numerator
    # ERsq_Pi_MSRR_new = (1+kappa_z_theory)**2 * (sigma_star**2 * c_M * Q/(sigma_star**2) \
    #                                             + (1+kappa_z_theory)**(-2) * (b_star1*(Gamma2_z + z*Gamma2_z_prime)/b) \
    #                                             - 2*c_M * sigma_star**2 * Q/(sigma_star**2)* (b*psi_star1 - z*Gamma1_z)/(k_star*b)*b_star1 )

    ERsq_Pi_MSRR_new = (1 + kappa_z_theory) ** 2 * (c_M*Q* (1-2*((b*psi_star1 - z*Gamma1_z)/(k_star * b))*b_star1)\
                                                     + (1+kappa_z_theory)**(-2) * b_star1 * (Gamma2_z + z*Gamma2_z_prime)/b)

    # Expected SRU by Prop 13
    Expected_SRU_MSRR_new_numerator = sigma_star ** 2 * (b_star0 * psi_star1 - z * N * Gamma1_z) / (k_star*(1 + kappa_z_theory))
    Expected_SRU_MSRR_new_denominator = (c_M*Q* (1-2*((b*psi_star1 - z*Gamma1_z)/(k_star * b))*b_star1)\
                                                     + (1+kappa_z_theory)**(-2) * b_star1 * (Gamma2_z + z*Gamma2_z_prime)/b) ** (1 / 2)
    Expected_SRU_MSRR_new = Expected_SRU_MSRR_new_numerator / Expected_SRU_MSRR_new_denominator

    return ERsq_Pi_MSRR_new, ERsq_Pi_MSRR_factor1, ERsq_Pi_MSRR_factor2, ERsq_Pi_MSRR_factor3, Gamma0_z, Gamma1_z, Gamma2_z,\
           Gamma3_z, Gamma4_z, Gamma1_z_prime, Gamma2_z_prime, Expected_SRU_MSRR_new


def benchmark_SRU(M, norm_beta_squared, Sigma, Psi):
    """
    The function computes the benchmark SRU, which is the case of z = c = 0 of function expected_squared_return_ols
    :param M: number of variables
    :param norm_beta_squared: the level of beta, b = ||beta||^2
    :param Sigma: covariance of signals across assets
    :param Psi: covariance across signals
    :return: benchmark SRU
    """
    # numerator
    numerator = np.trace(Sigma) * norm_beta_squared/M*np.trace(Psi)
    # denominator
    denominator_term1 = norm_beta_squared ** 2 * ((np.trace(Sigma))**2 + np.trace(Sigma**2)) * ((np.trace(Psi)/M)**2)
    denominator_term2 = (np.trace(Sigma) + norm_beta_squared/M*np.trace(Psi)*np.trace(Sigma**2))*(norm_beta_squared /M *np.trace(Psi))
    denominator = np.sqrt(denominator_term1 + denominator_term2)
    return numerator / denominator

# def benchmark_SRU2(M, norm_beta_squared, Sigma, Psi):
#     """
#     The function computes the benchmark SRU, which is the case of z = c = 0 of function expected_squared_return_ols
#     :param M: number of variables
#     :param norm_beta_squared: the level of beta, b = ||beta||^2
#     :param Sigma: covariance of signals across assets
#     :param Psi: covariance across signals
#     :return: benchmark SRU
#     """
#     # numerator
#     numerator = np.trace(Sigma) * norm_beta_squared/M*np.trace(Psi)
#     # denominator
#     denominator_term1 = norm_beta_squared ** 2 * ((np.trace(Sigma))**2 + np.trace(np.linalg.matrix_power(Sigma,2))) * ((np.trace(Psi)/M)**2)
#     denominator_term2 = (np.trace(Sigma) + norm_beta_squared/M*np.trace(Psi)*np.trace(np.linalg.matrix_power(Sigma,2)))*(norm_beta_squared /M *np.trace(Psi))
#     denominator = np.sqrt(denominator_term1 + denominator_term2)
#     return numerator / denominator


def Herfindal(Sigma):
    """
    Herfindal of benchmark_SRU(M, norm_beta_squared, Sigma, Psi)
    :param Sigma: covariance of signals across assets
    :return: Herfindal
    """
    return (np.trace(Sigma))**2/(np.trace(np.linalg.matrix_power(Sigma,2)))
    # (np.trace(Sigma))**2/(np.trace(Sigma**2))





