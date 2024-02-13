# This file contains functions for 6. Eigenvalue Distributions.
import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import special
from expectation_functions import marcenko_pastur
import matplotlib.pyplot as plt
from Psi_Sigma_Sigma_eps_functions import *

############ Power distribution ############
def normalization_constant(alpha_, gamma_):
    """
    This function computes normalization constant c
    :param alpha_:
    :param gamma_:
    :return:
    """
    return 1/((alpha_ + 1) ** (-gamma_-1) * special.gamma(gamma_ + 1))
    # check c: prove c = c_check
    # integral_func = lambda x: x**alpha_ * (np.abs(np.log(x)))**gamma_
    # c_check = 1/integrate.quad(integral_func, 0, 1)[0]
    # print((c_check-c)/c)


def eigenvalue_distribution(lambda_, alpha_, gamma_):
    """
    This function defines the PDF of eigenvalues, p(lambda_, alpha_, gamma_), p(x) in section 6 Eigenvalue Distributions in Sims20210615.PDF
    :param lambda_:
    :param alpha_: alpha_>-1
    :param gamma_: gamma_>-1
    :return: p(lambda_, alpha_, gamma_) = c*(lambda_**alpha_)* np.abs(np.log(lambda_))**gamma_
    """
    # kappa is a normalization constant
    kappa = normalization_constant(alpha_, gamma_)
    return kappa * (lambda_ ** alpha_) * (np.abs(np.log(lambda_)) ** gamma_)
    # test: the PDF sums up to 1
    # step = 0.0001
    # x = np.arange(step, 1, step)
    # eigen = eigenvalue_distribution(x, 1, 1)
    # sum(eigen)*step # = 1

def integrand_f(x, alpha_Psi, gamma_Psi,m):
    """
    This function is the integrand of f = F(m) - z, which is also the integrand of F(m)
    :param x: eigen value
    :param alpha_Psi: alpha of Psi
    :param gamma_Psi: gamma of Psi
    :param m: the value of Marcenko-Pastur Theorem m(-z, c), which is the variable to solve
    :return: integrand of f
    """
    return eigenvalue_distribution(x, alpha_Psi, gamma_Psi) * x / (1 + m * x)


def integrand_fprime(x, alpha_Psi, gamma_Psi, m):
    """
    This function is the integrand of f' = F'(m), which is used for Newton's method
    :param x: eigen value
    :param alpha_Psi: alpha of Psi
    :param gamma_Psi: gamma of Psi
    :param m: the value of Marcenko-Pastur Theorem m(-z, c), which is the variable to solve
    :return: integrand of f'
    """
    return eigenvalue_distribution(x, alpha_Psi, gamma_Psi) * (x ** 2) / ((1+m*x)**2)


def f_function(m, alpha_Psi, gamma_Psi, c, z):
    """
    f(m, z) = F(m) - z
    :param m: the value of Marcenko-Pastur Theorem m(-z, c), which is the variable to solve
    :param z: the value of F(m)
    :return: F(m) - z
    """
    return -1/m + c * integrate.quad(integrand_f, 0, 1, args=(alpha_Psi, gamma_Psi, m))[0] - z


def fprime_function(m, alpha_Psi, gamma_Psi, c, z):
    """
    f'(m, z) = F'(m)
    :param m: the value of Marcenko-Pastur Theorem m(-z, c), which is the variable to solve
    :param alpha_Psi: alpha of Psi
    :param gamma_Psi: gamma of Psi
    :param c: constant
    :param z: the value of F(m)
    :return: F'(m)
    """
    return 1 / (m ** 2) - c * integrate.quad(integrand_fprime, 0, 1, args=(alpha_Psi, gamma_Psi, m))[0]


def solve_m_Theorem3(alpha_Psi, gamma_Psi, x, z, c):
    """
    This function solves tilde{m} from F(m) = z, which is also f(m,z) = F(m) - z = 0
    :param alpha_Psi: alpha of Psi
    :param gamma_Psi: gamma of Psi
    :param x: initial estimate of the m
    :param z: the value of F(m)
    :param c: constant, set as c = M/(N*T)
    :return:
    """
    loop_res = [optimize.newton(f_function, x0, fprime=fprime_function,
                                args=(alpha_Psi, gamma_Psi, c, z0)) for x0, z0 in zip(x, z)]
    return loop_res


def integrand_stieltjes(x, alpha_Psi, gamma_Psi, z):
    """
    careful: only defined for negative z!!
    :param x:
    :param alpha_Psi:
    :param gamma_Psi:
    :param z:
    :return:
    """
    return eigenvalue_distribution(x, alpha_Psi, gamma_Psi) / (x - z)


def stiejties_transform_of_power_law(alpha_Psi, gamma_Psi, z): # get
    """
    Stieltjes transform of the power law distribution by Eq. (10)
    :param alpha_Psi:
    :param gamma_Psi:
    :return:
    """
    return integrate.quad(integrand_stieltjes, 0, 1, args=(alpha_Psi, gamma_Psi, z))[0]


def map_from_m_to_mPsi(m, z, c):
    """
    This function maps m to m_Psi by Eq. (12)
    :param m: estimated \hat{m} by observations
    :param z: the value of F(tilde_m)
    :param c: constant, set as c = M/(N*T)
    :return:
    """
    return (1 - c - c*z*m)**2*m/z


def map_from_tilde_m_to_m(tilde_m, z, c):
    """
    This function maps m to tilde{m} by Eq. (13)
    :param tilde_m: tilde_m is the solution of solve_m_Theorem3(alpha_Psi, gamma_Psi, x, z, c)
    :param z: the value of F(tilde_m)
    :param c: constant, set as c = M/(N*T)
    :return:
    """
    return ((1 - c) / z + tilde_m) / c


def plot_x_vs_tilde_m(z, x, tilde_m_of_minus_z, c):
    """
    Plot m_Psi(z) from Eq. (10) vs. tilde m solved from Theorem3
    :param z: a vector of ridge penalty parameter
    :param x: m_Psi(z) from Eq. (10), x = stiejties_transform_of_power_law(alpha_Psi, gamma_Psi, - z)
    :param tilde_m_of_minus_z: tilde m solved from Theorem3, tilde_m_of_minus_z = solve_m_Theorem3(alpha_Psi, gamma_Psi, x, z, c)
    :return: Plot
    """
    plot_path = '/Users/kyzhou/Dropbox/MSRR/Code/Output/Output_6_Eigenvalue Distributions'
    fig = plt.figure()
    plt.plot(z, x, color='red', linewidth=4, label='m_Psi(z) from Eq. (10)')
    plt.plot(z, tilde_m_of_minus_z, color='blue', linewidth=2, label='tilde m from Theorem3')
    plt.xlabel('z')
    plt.ylabel('tilde m')
    plt.title('Check Theorem 3, m_Psi vs. tilde m, c = '+str(c))
    plt.legend()
    fig.savefig(plot_path + '/Theorem 3 check, m_Psi vs. tilde m, c = '+str(c) +'.png')
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def plot_x_vs_m(z, x, m, c):
    """
    Plot m_Psi(z) from Eq. (10) vs. m solved from Theorem3
    :param z: a vector of ridge penalty parameter
    :param x: m_Psi(z) from Eq. (10), x = stiejties_transform_of_power_law(alpha_Psi, gamma_Psi, - z)
    :param m: m from from Theorem3, m = map_from_tilde_m_to_m(tilde_m, z, c)
    :param c: constant, set as c = M/(N*T)
    :return: Plot
    """
    plot_path = '/Users/kyzhou/Dropbox/MSRR/Code/Output/Output_6_Eigenvalue Distributions'
    fig = plt.figure()
    plt.plot(z, x, color='red', linewidth=4, label='m_Psi(z) from Eq. (10)')
    plt.plot(z, m, color='blue', linewidth=2, label='m from Theorem3')
    plt.xlabel('z')
    plt.ylabel('m')
    plt.title('Check Theorem 3, m_Psi vs. m, c = ' + str(c))
    plt.legend()
    fig.savefig(plot_path + '/Theorem 3 check, m_Psi vs. m, c = ' + str(c) + '.png')
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def plot_mMarcenkoPastur_vs_m(z, m_MarcenkoPastur, m, c):
    """
    Plot m(-z) from Marcenko-Pastur Theorem vs. m solved from Theorem3
    :param z: a vector of ridge penalty parameter
    :param m_MarcenkoPastur: m(-z) from Marcenko-Pastur Theorem, m_MarcenkoPastur = marcenko_pastur(1, c, z)
    :param m: m from from Theorem3, m = map_from_tilde_m_to_m(tilde_m, z, c)
    :param c: constant, set as c = M/(N*T)
    :return: Plot
    """

    plot_path = '/Users/kyzhou/Dropbox/MSRR/Code/Output/Output_6_Eigenvalue Distributions'
    fig = plt.figure()
    plt.plot(z, m_MarcenkoPastur, color='red', linewidth=4, label='m(-z) from Marcenko-Pastur Theorem')
    plt.plot(z, m, color='blue', linewidth=2, label='m from Theorem3')
    plt.xlabel('z')
    plt.ylabel('m')
    plt.title('Check Theorem 3, m_MarcenkoPastur vs. m, c = ' + str(c))
    plt.legend()
    fig.savefig(plot_path + '/Theorem 3 check, m_MarcenkoPastur vs. m, c = ' + str(c) + '.png')
    plt.show(block=False)
    plt.pause(1)
    plt.close()

def Stieltjes_transform_Theorem3(alpha_Psi, gamma_Psi, z, c):
    # m_Psi(z) from Eq. (10), used for initialization of solve_m_Theorem3(alpha_Psi, gamma_Psi, x, -z, c)
    x = [stiejties_transform_of_power_law(alpha_Psi, gamma_Psi, - z_) for z_ in z]

    # tilde m solved from Theorem3
    tilde_m_of_minus_z = solve_m_Theorem3(alpha_Psi, gamma_Psi, x, -z, c)

    # m solved from Theorem3, m should be equal to x
    m = map_from_tilde_m_to_m(tilde_m_of_minus_z, - z, c)
    return m

################ real Psi eigenvalues ################
def m_tilde_initialization_upper_bound(a_star, c, psi_eigenvalues):
    '''
    This function initialize for function solve_m_Theorem3_psi_eigenvalues
    :param a_star:
    :param z:
    :param psi_eigenvalues:
    :return:
    '''
    m_tilde_z = ((np.min(a_star*psi_eigenvalues)*(np.sqrt(c) - 1)) ** (-1))
    return m_tilde_z

def m_Psi_z(a_star, z, M, psi_eigenvalues):
    '''
    This function computes m_Psi(z) from Section 4, simulation. It's used for initialization of solve_m_Theorem3(alpha_Psi, gamma_Psi, x, -z, c)
    :param a_star: a_star = sigma_star = np.trace(sigma_hat)/N
    :param z:
    :param M:
    :param psi_eigenvalues:
    :return:
    '''
    m_Psi_z = [1/M*np.sum((a_star*psi_eigenvalues - z_)**(-1)) for z_ in z]
    return m_Psi_z

def f_function_psi_eigenvalues(m_tilde, c, z, a_star, M, psi_eigenvalues):
    """
    f(m, z) = F(m) - z
    :param m: the value of Marcenko-Pastur Theorem m(-z, c), which is the variable to solve
    :param z: the value of F(m)
    :return: F(m) - z
    """
    return 1/m_tilde - c/M*np.sum(a_star*psi_eigenvalues/(1 + m_tilde*a_star*psi_eigenvalues)) - z

def F_tildem_psi_eigenvalues(m_tilde, c, a_star, M, psi_eigenvalues):
    """
    F(m) in Section 4
    :param m: the value of Marcenko-Pastur Theorem m(-z, c), which is the variable to solve
    :param z: the value of F(m)
    :return: F(m) - z
    """
    factor2 = np.array([c/M*np.sum(a_star*psi_eigenvalues/(1 + m_tilde_i*a_star*psi_eigenvalues)) \
                        for m_tilde_i in m_tilde])
    return 1/m_tilde - factor2


def fprime_function_psi_eigenvalues(m_tilde, c, z, a_star, M, psi_eigenvalues):
    """
    f'(m, z) = F'(m)
    :param m: the value of Marcenko-Pastur Theorem m(-z, c), which is the variable to solve
    :param alpha_Psi: alpha of Psi
    :param gamma_Psi: gamma of Psi
    :param c: constant
    :param z: the value of F(m)
    :return: F'(m)
    """
    return  -1 / (m_tilde ** 2) + c/M *np.sum((a_star*psi_eigenvalues/(1 + m_tilde*a_star*psi_eigenvalues))**2)

def solve_m_Theorem3_psi_eigenvalues(x, c, z, a_star, M, psi_eigenvalues):
    """
    This function solves tilde{m} from F(m) = z with true psi_eigenvalues, which is also f(m,z) = F(m) - z = 0
    :param x: initial estimate of the m
    :param z: the value of F(m)
    :param c: constant, set as c = M/(N*T)
    :return:
    """
    loop_res = [optimize.newton(f_function_psi_eigenvalues, x0, fprime=fprime_function_psi_eigenvalues,
                                args=(c, z0, a_star, M, psi_eigenvalues)) for x0, z0 in zip(x, z)]
    return loop_res

def Stieltjes_transform_Theorem3_psi_eigenvalues(z, c, a_star, M, psi_eigenvalues):
    # m_Psi(z) from Eq. (10), used for initialization of solve_m_Theorem3(alpha_Psi, gamma_Psi, x, -z, c)
    x = m_Psi_z(a_star, -z, M, psi_eigenvalues)
    # x = m_tilde_initialization_upper_bound(a_star, c, psi_eigenvalues)

    # tilde m solved from Theorem3
    tilde_m_of_minus_z = solve_m_Theorem3_psi_eigenvalues(x, c, z, a_star, M, psi_eigenvalues)

    # m solved from Theorem3, m should be equal to x
    m = map_from_tilde_m_to_m(tilde_m_of_minus_z, z, c)
    return m



if __name__ == '__main__':
    # Check: when c -> 0, z should be similar to F_value

    ############ Power distribution ############
    # Parameters setting
    c = 0.001 # 5 # M/(N*T)
    alpha_Psi = 1
    gamma_Psi = 1
    z = np.arange(0.2, 10, 0.05)

    # m_Psi(z) from Eq. (10), used for initialization of solve_m_Theorem3(alpha_Psi, gamma_Psi, x, -z, c)
    x = [stiejties_transform_of_power_law(alpha_Psi, gamma_Psi, - z_) for z_ in z]

    # tilde m solved from Theorem3
    tilde_m_of_minus_z = solve_m_Theorem3(alpha_Psi, gamma_Psi, x, -z, c)

    # m solved from Theorem3, m should be equal to x
    m = map_from_tilde_m_to_m(tilde_m_of_minus_z, - z, c)

    # m(-z) from Marcenko-Pastur Theorem
    m_MarcenkoPastur = marcenko_pastur(1, c, z)

    # Plot to check
    plot_x_vs_tilde_m(z, x, tilde_m_of_minus_z, c)
    plot_x_vs_m(z, x, m, c)
    plot_mMarcenkoPastur_vs_m(z, m_MarcenkoPastur, m, c)

    ############ Real Psi distribution ############
    M = 500
    N = 100
    T = 100
    c = M / (N * T)
    c_M = M/T

    # Parameters for Sigma generation
    a_Sigma = 2  # [0.5, 1, 2]
    # Parameters for Psi generation
    a_Psi = 2  # [0.5, 1, 2]
    # Parameters for Sigma_eps generation
    a_Sigma_eps = 0  # [0, 0.1, 0.5, 1, 2]

    # Sigma (covariance of signals across assets) generation
    Sigma, sigma_eigenvalues = generate_Sigma_with_empirical_eigenvalues(N, a_Sigma)

    # Psi (covariance across signals) generation
    Psi, psi_eigenvalues = generate_Psi_with_empirical_eigenvalues(M, a_Psi)

    # Sigma_eps generation
    Sigma_eps = generate_Sigma_epsilon(N, a_Sigma_eps)

    # Sigma_hat and its eigenvalues
    sigma_hat, sigma_hat_eigenvalues = sigma_hat_function(Sigma, Sigma_eps)

    sigma_star = np.sum(sigma_hat_eigenvalues) / N

    m_tilde = np.arange(0.01, 100, 0.01)

    # For MV
    F = F_tildem_psi_eigenvalues(m_tilde, c, sigma_star, M, psi_eigenvalues)
    F_min = min(F) # 0.009992786913449722
    F_max = max(F) # 99.99937618244566

    z_index = (z>F_min) & (z<F_max)
    z = z[z_index]
    print(len(z))

    m = Stieltjes_transform_Theorem3_psi_eigenvalues(z, c, sigma_star, M, psi_eigenvalues)
    print('Successfully computed m in MV')

    # f_function_psi_eigenvalues(m_tilde, c, z[10], a_star, M, psi_eigenvalues)
    # m_test = np.array([f_function_psi_eigenvalues(m_tilde[i], c, z[10], a_star, M, psi_eigenvalues) for i in
    #                    range(len(m_tilde))])
    # tilde_m_of_minus_z = solve_m_Theorem3_psi_eigenvalues(x[0:59], c, z[0:59], a_star, M, psi_eigenvalues)
    #
    # tilde_m_of_minus_z = solve_m_Theorem3_psi_eigenvalues(list(x * np.ones(len(z_test))), c, z_test, a_star, M,
    #                                                       psi_eigenvalues)
    #

    # For MSRR
    F_cM = F_tildem_psi_eigenvalues(m_tilde, c_M, sigma_star, M, psi_eigenvalues)
    F_cM_min = min(F_cM)  # -0.12878060000928507
    F_cM_max = max(F_cM)  # 96.88091222829783

    z_index = (z > F_cM_min) & (z < F_cM_max)
    z = z[z_index]
    print(len(z)) # [2:len(z)]
    q_minus_z = Stieltjes_transform_Theorem3_psi_eigenvalues(z, c_M, sigma_star, M,
                                                             psi_eigenvalues)  # marcenko_pastur(1, c_M, z) # Prop 12
    # z[0:1] doesn't work
    print('Successfully computed q_minus_z in MSRR')




    # fig, ax = plt.subplots()
    # # make a plot
    # l1 = plt.plot(m_tilde, F_cM, color='red',
    #          linestyle='solid', linewidth=3, label = 'MSRR SRU')




