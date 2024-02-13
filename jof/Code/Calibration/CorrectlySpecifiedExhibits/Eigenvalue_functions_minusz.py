# This file contains functions for 6. Eigenvalue Distributions.
import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import special
from expectation_functions import marcenko_pastur
import matplotlib.pyplot as plt
from Psi_Sigma_Sigma_eps_functions import *
import os.path
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.setrecursionlimit(5000)

# if 'kz272' in os.path.expanduser('~'):
#     code_path = '/gpfs/loomis/home.grace/kz272/scratch60/MSRR/Code/Sims20210615_Python/Grace_MSRR_Output'
# elif 'kyzhou' in os.path.expanduser('~'):
#     code_path = '/Users/kyzhou/Dropbox/MSRR/Code/Output'
# elif 'malamud' in os.path.expanduser('~'):
#     code_path = '/Users/malamud/Dropbox/MY_STUFF/RESEARCH/MSRR/Code/SemyonOutput'
# else:
#     code_path = 'SemyonOutput'
#
# if not os.path.exists(code_path):
#     os.mkdir(code_path)

def bisect(fn, c, z, a_star, M, psi_eigenvalues,  a, b, level, precision, maxiter):
    """
    bisection method for solving fn(x)=level on [a,b]
    """
    # precision = 0.00001
    # print('max iter = %s' %(maxiter))
    count = 0
    value = fn(a, c, z, a_star, M, psi_eigenvalues) - level
    if (fn(a, c, z, a_star, M, psi_eigenvalues) - level) * (fn(b,  c, z, a_star, M, psi_eigenvalues) - level) > 10 ** (- 10):
        raise Exception('No change of sign - bisection not possible')
    while np.abs(value) > precision and count < maxiter:
        count += 1
        x = 0.5 * (a + b)
        if (fn(x,  c, z, a_star, M, psi_eigenvalues) - level) * (fn(b,  c, z, a_star, M, psi_eigenvalues) - level) <= 0:
            a = x
            value, a = bisect(fn,  c, z, a_star, M, psi_eigenvalues, a, b, level, precision, maxiter)
        else:
            b = x
            value, a = bisect(fn,  c, z, a_star, M, psi_eigenvalues, a, b, level, precision, maxiter)
    return value, a

# def bisect(fn, c, z, a_star, M, psi_eigenvalues, a, b, level, precision, maxiter):
#     '''
#     bisection method for solving fn(x)=level on [a,b]
#     :param fn:
#     :param y:
#     :param a:
#     :param b:
#     :param level:
#     :param precision:
#     :param maxiter:
#     :return:
#     '''
#     # precision = 0.00001
#     count = 0
#     value = fn(a, c, z, a_star, M, psi_eigenvalues) - level
#     if (fn(a, c, z, a_star, M, psi_eigenvalues) - level) * (fn(b, c, z, a_star, M, psi_eigenvalues) - level) > 10 ** (- 10):
#         raise Exception('No change of sign - bisection not possible')
#     while np.abs(value) > precision and count < maxiter:
#         count += 1
#         x = 0.5 * (a + b)
#         if (fn(x, c, z, a_star, M, psi_eigenvalues) - level) * (fn(b, c, z, a_star, M, psi_eigenvalues) - level) <= 0:
#             a = x
#             value, a = bisect(fn, c, z, a_star, M, psi_eigenvalues, a, b, level, precision, maxiter)
#         else:
#             b = x
#             value, a = bisect(fn, c, z, a_star, M, psi_eigenvalues, a, b, level, precision, maxiter)
#     return value, a

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
    loop_res = [optimize.newton(f_function, x0, fprime=fprime_function, tol = 1e-12, maxiter=500,
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
    return (-(1 - c) / z + tilde_m) / c


def plot_x_vs_tilde_m(z, x, tilde_m_of_minus_z, c):
    """
    Plot m_Psi(z) from Eq. (10) vs. tilde m solved from Theorem3
    :param z: a vector of ridge penalty parameter
    :param x: m_Psi(z) from Eq. (10), x = stiejties_transform_of_power_law(alpha_Psi, gamma_Psi, - z)
    :param tilde_m_of_minus_z: tilde m solved from Theorem3, tilde_m_of_minus_z = solve_m_Theorem3(alpha_Psi, gamma_Psi, x, z, c)
    :return: Plot
    """
    plot_path = '/Users/kyzhou/Dropbox/MSRR/Code/Output/Output_6_Eigenvalue Distributions'
    # fig = plt.figure()
    # plt.plot(z, x, color='red', linewidth=4, label='m_Psi(z) from Eq. (10)')
    # plt.plot(z, tilde_m_of_minus_z, color='blue', linewidth=2, label='tilde m from Theorem3')
    # plt.xlabel('z')
    # plt.ylabel('tilde m')
    # plt.title('Check Theorem 3, m_Psi vs. tilde m, c = '+str(c))
    # plt.legend()
    # fig.savefig(plot_path + '/Theorem 3 check, m_Psi vs. tilde m, c = '+str(c) +'.png')
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close()

    fig, ax = plt.subplots()
    # make a plot
    l1 = plt.plot(z, x, color='red', linewidth=4, label=r'$m_\Psi(z)$ from Eq. (10)')
    l2 = plt.plot(z, tilde_m_of_minus_z, color='blue', linewidth=2, label=r'$\tilde{m}$ from Theorem3')
    ax.set_xlabel("z")
    # set y-axis label
    ax.set_ylabel('Value', color="red", fontsize=12)

    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    l3 = ax2.plot(z, np.array(tilde_m_of_minus_z) / np.array(x), color='black', linestyle='solid', linewidth=1.5,
                  label=r'$\tilde{m}/m_\Psi(z)$')  # 1.04442995
    # l5 = ax2.plot(z, Gamma1_T_hat / Gamma1_T_simu, color='black', linestyle='dashed', linewidth=1,
    #               label='Lemma 29/Simulated')
    ax2.set_ylabel("Quotient", color="black", fontsize=12)

    # added these three lines and add legend
    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    plt.title(r'Check Theorem 3, $m_\Psi$ vs. $\tilde{m}$, $c$ = '+str(c))
    plt.rcParams["axes.titlesize"] = 10
    plt.tight_layout()
    fig.savefig(plot_path + '/Theorem 3 check, m_Psi vs. tilde m, c = '+str(c) +'.png')
    plt.show(block=False)
    plt.pause(1)
    plt.close('all')


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
    # fig = plt.figure()
    # plt.plot(z, x, color='red', linewidth=4, label='m_Psi(z) from Eq. (10)')
    # plt.plot(z, m, color='blue', linewidth=2, label='m from Theorem3')
    # plt.xlabel('z')
    # plt.ylabel('m')
    # plt.title('Check Theorem 3, m_Psi vs. m, c = ' + str(c))
    # plt.legend()
    # fig.savefig(plot_path + '/Theorem 3 check, m_Psi vs. m, c = ' + str(c) + '.png')
    # plt.show(block=False)
    # plt.pause(1)
    # plt.close()

    fig, ax = plt.subplots()
    # make a plot
    l1 = plt.plot(z, x, color='red', linewidth=4, label=r'$m_\Psi(z)$ from Eq. (10)')
    l2 = plt.plot(z, m, color='blue', linewidth=2, label=r'$m$ from Theorem3')
    ax.set_xlabel("z")
    # set y-axis label
    ax.set_ylabel('Value', color="red", fontsize=12)

    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    l3 = ax2.plot(z, np.array(m) / np.array(x), color='black', linestyle='solid', linewidth=1.5,
                  label=r'$m/m_\Psi(z)$')  # 1.04442995
    # l5 = ax2.plot(z, Gamma1_T_hat / Gamma1_T_simu, color='black', linestyle='dashed', linewidth=1,
    #               label='Lemma 29/Simulated')
    ax2.set_ylabel("Quotient", color="black", fontsize=12)

    # added these three lines and add legend
    lns = l1 + l2 + l3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)

    plt.title(r'Check Theorem 3, $m_\Psi$ vs. $m$, $c$ = ' + str(c))
    plt.rcParams["axes.titlesize"] = 10
    plt.tight_layout()
    fig.savefig(plot_path + '/Theorem 3 check, m_Psi vs. m, c = ' + str(c) + '.png')
    plt.show(block=False)
    plt.pause(1)
    plt.close('all')

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
    m_tilde_z = ((np.max(a_star*psi_eigenvalues)*(np.sqrt(c) - 1)) ** (-1))
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

def derivative_of_m_Psi_z(a_star, z, M, psi_eigenvalues):
    '''
    This function computes m_Psi'(z) by taking the first derivative of Eq. (10)
    :param a_star: a_star = sigma_star = np.trace(sigma_hat)/N
    :param z:
    :param M:
    :param psi_eigenvalues:
    :return:
    '''
    derivative_of_m_Psi_z = [1/M*np.sum((a_star*psi_eigenvalues - z_)**(-2)) for z_ in z]
    return derivative_of_m_Psi_z


def f_function_psi_eigenvalues(m_tilde, c, z, a_star, M, psi_eigenvalues):
    """
    f(m, z) = F(m) - z
    :param m: the value of Marcenko-Pastur Theorem m(-z, c), which is the variable to solve
    :param z: the value of F(m)
    :return: F(m) - z
    """
    return 1/m_tilde - c/M*np.sum(a_star*psi_eigenvalues/(1 + m_tilde*a_star*psi_eigenvalues)) - z

# # test
# m_tilde = np.arange(0.01, 100, 0.01)
# F = np.array([f_function_psi_eigenvalues(m_tilde[i], c_M, -z0, a_star, M, psi_eigenvalues) for i in range(len(m_tilde))])
# fig, ax = plt.subplots()
# l1 = plt.plot(m_tilde, F, color='blue', linestyle='solid', linewidth=1, label='f')
# plt.close('all')


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

# # test
# m_tilde = np.arange(10000,100000000, 10000)
# F = F_tildem_psi_eigenvalues(m_tilde, c, a_star, M, psi_eigenvalues)
# fig, ax = plt.subplots()
# l1 = plt.plot(m_tilde, F, color='blue', linestyle='dashed', linewidth=3, label='OLS Expectation')

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
    return  - 1 / (m_tilde ** 2) + c/M *np.sum((a_star*psi_eigenvalues/(1 + m_tilde*a_star*psi_eigenvalues))**2)

# # test
# # build save path
# save_path = os.path.join(code_path, 'check_Newton_F')
# if not os.path.exists(save_path):
#     os.mkdir(save_path)
#
# para_str = 'M = ' + str(M) + ', N = ' + str(N) + ', T = ' + str(T) + ', T_test = ' + str(T_test) \
#                + ', N_sim = ' + str(N_sim) +'\n' + special_str
#
# m_tilde = np.arange(-10, 0, 0.1)
# F = F_tildem_psi_eigenvalues(m_tilde, c, a_star, M, psi_eigenvalues)
# F_prime = np.array([fprime_function_psi_eigenvalues(m_tilde_i, c, z, a_star, M, psi_eigenvalues) for m_tilde_i in m_tilde])
# fig, ax = plt.subplots()
# l1 = plt.plot(m_tilde, F, color='red', linestyle='solid', linewidth=2, label='F')
# l2 = plt.plot(m_tilde, F_prime, color='blue', linestyle='solid', linewidth=1, label='F\'')
# # set x-axis label
# ax.set_xlabel(r"$\tilde{m}$")
# # set y-axis label
# ax.set_ylabel('Value', color="red", fontsize=12)
# lns = l1 + l2
# labs = [l.get_label() for l in lns]
# ax.legend(lns, labs, loc=0)
#
# plt.title('Check Newton method \n ' + para_str)
# plt.rcParams["axes.titlesize"] = 10
# plt.tight_layout()
# fig.savefig(save_path + '/Check Newton method, ' + para_str + '.png')
# plt.show(block=False)
# plt.pause(1)
# plt.close('all')
#
# print(min(F))
# print(max(F))
# print(min(F_prime))
# print(max(F_prime))


def solve_m_Theorem3_psi_eigenvalues(x, c, z, a_star, M, psi_eigenvalues):
    """
    This function solves tilde{m} from F(m) = z with true psi_eigenvalues, which is also f(m,z) = F(m) - z = 0
    :param x: initial estimate of the m
    :param z: the value of F(m)
    :param c: constant, set as c = M/(N*T)
    :return:
    """
    loop_res = [optimize.newton(f_function_psi_eigenvalues, x0, fprime=fprime_function_psi_eigenvalues, tol = 1e-12, maxiter=10000000,
                                args=(c, z0, a_star, M, psi_eigenvalues)) for x0, z0 in zip(x, z)]
    return loop_res

    # # test
    # def f(x):
    #     return (x**2 - 2)
    #
    # root = optimize.newton(f, 1.5, fprime=lambda x: 2*x, tol = 1e-15, maxiter=500000)


def solve_m_Theorem3_psi_eigenvalues_bisect(c, z, a_star, M, psi_eigenvalues, lower_bound, upper_bound):
    """
    This function solves tilde{m} from F(m) = z with true psi_eigenvalues, which is also f(m,z) = F(m) - z = 0
    :param x: initial estimate of the m
    :param z: the value of F(m)
    :param c: constant, set as c = M/(N*T)
    :return:
    """
    precision = 1e-10
    maxiter = 3000
    level = 0
    loop_res = [bisect(f_function_psi_eigenvalues, c, z0, a_star, M, psi_eigenvalues, a, b, level, precision, maxiter)[1] \
                for a,b,z0 in zip(lower_bound, upper_bound, z)]
    return loop_res


def m_tilde_lower_bound(z, c, a_star, psi_star1):
    '''
    This function gives the lower bound of m_tilde by Appendix E
    :param z:
    :param c:
    :param a:
    :param psi_star1:
    :return:
    '''
    lower_bound = 1/(z + c*a_star*psi_star1)
    return lower_bound

def m_tilde_upper_bound(z, c, a_star, psi_eigenvalues, M):
    '''
    This function gives the upper bound of m_tilde by Appendix E
    :param z:
    :param c:
    :param a:
    :param psi_star1:
    :return:
    '''
    psi_tilde = 1/M*np.sum(1/(a_star*psi_eigenvalues))
    upper_bound_numerator = -(z*psi_tilde + c - 1) + np.sqrt((z*psi_tilde + c - 1)**2 + 4*z*psi_tilde)
    upper_bound_denominator = 2*z
    upper_bound = upper_bound_numerator/upper_bound_denominator
    return upper_bound

def Stieltjes_transform_Theorem3_psi_eigenvalues(z, c, a_star, M, psi_eigenvalues, psi_star1):
    '''
    This function computes the m(-z) by Theorem 3
    :param z:
    :param c:
    :param a_star:
    :param M:
    :param psi_eigenvalues:
    :return:
    '''
    # m_Psi(z) from Eq. (10), used for initialization of solve_m_Theorem3(alpha_Psi, gamma_Psi, x, -z, c)
    # x = m_Psi_z(a_star, -z, M, psi_eigenvalues)
    # x = 0.01*np.ones(len(z))
    # lower bound
    lower_bound = m_tilde_lower_bound(z, c, a_star, psi_star1)
    # upper bound
    upper_bound = m_tilde_upper_bound(z, c, a_star, psi_eigenvalues, M)

    # x = m_tilde_initialization_upper_bound(a_star, c, psi_eigenvalues)
    # tilde m solved from Theorem3
    tilde_m_of_z = solve_m_Theorem3_psi_eigenvalues(lower_bound, c, z, a_star, M, psi_eigenvalues)
    # tilde_m_of_z = solve_m_Theorem3_psi_eigenvalues_bisect(c, z, a_star, M, psi_eigenvalues, lower_bound, upper_bound)

    # check whether m_tilde is within the correct range
    lower_bound_check = np.sum(tilde_m_of_z >= lower_bound) == len(z)
    if lower_bound_check == False:
        print('Lower bound check is not passed, sum = %s'%(np.sum(tilde_m_of_z >= lower_bound)))
    else:
        print('Lower bound check is passed!')
    upper_bound_check = np.sum(tilde_m_of_z <= upper_bound) == len(z)
    if upper_bound_check == False:
        print('Upper bound check is not passed, sum = %s'%(np.sum(tilde_m_of_z <= upper_bound)))
    else:
        print('Upper bound check is passed!')

    # m solved from Theorem3, m should be equal to x
    m_minus_z = map_from_tilde_m_to_m(tilde_m_of_z, z, c)
    return lower_bound, tilde_m_of_z, m_minus_z


def derivative_of_Stieltjes_transform_Theorem3_psi_eigenvalues(z, c, a_star, M, psi_eigenvalues, m_minus_z):
    '''
    This function computes the first derivative of m(-z) according to Eq. (55)
    :param z:
    :param c:
    :param a_star:
    :param M:
    :param psi_eigenvalues:
    :param m_minus_z:
    :return:
    '''
    derivative_of_m_minus_z_nominator = np.array([1/M * np.sum((1 + a_star*psi_eigenvalues*c*m_minus_z[z_idx])/((a_star*psi_eigenvalues\
                                        *(1-c+c*z[z_idx]*m_minus_z[z_idx]) + z[z_idx])**2)) for z_idx in range(len(z))])
    derivative_of_m_minus_z_denominator = np.array([1 + 1 / M * np.sum((a_star * psi_eigenvalues * c * z[z_idx]) / (
                (a_star * psi_eigenvalues * (1 - c + c * z[z_idx] * m_minus_z[z_idx]) + z[z_idx]) ** 2)) for z_idx in range(len(z))])
    derivative_of_m_minus_z = derivative_of_m_minus_z_nominator/derivative_of_m_minus_z_denominator
    return derivative_of_m_minus_z




def Check_Theorem3_Eq12(m_minusz, z, c, a_star, M, psi_eigenvalues):
    Eq12_LHS = m_minusz
    m_aPsi_var = -z / (1 - c + c * z * m_minusz)
    m_aPsi = m_Psi_z(a_star, m_aPsi_var, M, psi_eigenvalues)
    Eq12_RHS = 1 / (1 - c + c * z * m_minusz)* m_aPsi
    Eq12_quotient = Eq12_RHS / Eq12_LHS
    return Eq12_quotient, m_aPsi, Eq12_RHS

def plot_F_tildem(m_tilde, F, F_cM):
    '''
    This function plots the shape of $F(\tilde{m})$
    :param m_tilde:
    :param F:
    :param F_cM:
    :return:
    '''
    fig, ax = plt.subplots()
    # make a plot
    l1 = plt.plot(m_tilde, F, color='red',
                  linestyle='solid', linewidth=3, label=r'$F(\tilde{m})$, c = %s' % (c))
    l2 = plt.plot(m_tilde, F_cM, color='blue',
                  linewidth=1.5, label=r'$F(\tilde{m})$, c_M = %s' % (c_M))
    ax.set_xlabel("z")
    # set y-axis label
    ax.set_ylabel(r'$F(\tilde{m})$', color="red", fontsize=12)

    lns = l1 + l2  # + l3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    plt.title(r'Check $F(\tilde{m})$')
    plt.rcParams["axes.titlesize"] = 10
    plt.tight_layout()
    save_path = '/Users/kyzhou/Dropbox/MSRR/Code/Output/Output_6_Eigenvalue Distributions'
    fig.savefig(save_path + '/Check F_tildem.png')
    plt.show(block=False)
    plt.pause(1)
    plt.close('all')


def plot_Eq12_check(z, q_minus_z_MSRR, Eq12_RHS_c_M, m_aPsi_c_M, Eq12_quotient_c_M, c_M):
    '''
    This function checks the LHS and RHS of Eq. (12) about m(-z, c, a)
    :param z:
    :param q_minus_z_MSRR:
    :param Eq12_RHS_c_M:
    :param m_aPsi_c_M:
    :param Eq12_quotient_c_M:
    :return:
    '''
    fig, ax = plt.subplots()
    l1 = plt.plot(z, q_minus_z_MSRR, color='blue',
                  linewidth=3, label=r'Eq. (12) LHS, m(-z, c, a)')
    l2 = plt.plot(z, Eq12_RHS_c_M, color='green',
                  linewidth=1,
                  label=r'Eq. (12) RHS, $\frac{1}{1 - c+ czm(-z; c, a)}m_{a \Psi}(\frac{z}{1 - c+ czm(-z; c, a)})$')
    l3 = plt.plot(z, m_aPsi_c_M, color='red',
                  linestyle='solid', linewidth=2, label=r'$m_{a \Psi}(\frac{z}{1 - c+ czm(-z; c, a)})$')
    ax.set_xlabel("z")
    # set y-axis label
    ax.set_ylabel('Value', color="red", fontsize=12)

    # twin object for two different y-axis on the sample plot
    ax2 = ax.twinx()
    # make a plot with different y-axis using second axis object
    l4 = ax2.plot(z, Eq12_quotient_c_M, color='black', linestyle='solid', linewidth=1.5,
                  label='Quotient of Eq. (12) RHS/LHS')  # 1.04442995
    # l5 = ax2.plot(z, Gamma1_T_hat / Gamma1_T_simu, color='black', linestyle='dashed', linewidth=1,
    #               label='Lemma 29/Simulated')
    ax2.set_ylabel("Quotient", color="black", fontsize=12)

    lns = l1 + l2 + l3 + l4
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0)
    plt.title('Check Eq. (12), c = %s' % (c_M))
    plt.rcParams["axes.titlesize"] = 10
    plt.tight_layout()
    save_path = '/Users/kyzhou/Dropbox/MSRR/Code/Output/Output_6_Eigenvalue Distributions'
    fig.savefig(save_path + '/Check Eq. (12), c = %s.png' % (c_M))
    plt.show(block=False)
    plt.pause(1)
    plt.close('all')

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

    # # Plot to check
    # plot_x_vs_tilde_m(z, x, tilde_m_of_minus_z, c)
    # plot_x_vs_m(z, x, m, c)
    # plot_mMarcenkoPastur_vs_m(z, m_MarcenkoPastur, m, c)

    ############ Real Psi distribution ############
    M = 300
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

    m_tilde = np.arange(0.01, 2, 0.01)

    # For MV
    F = F_tildem_psi_eigenvalues(m_tilde, c, sigma_star, M, psi_eigenvalues)
    F_min = min(F) # 0.009992786913449722
    F_max = max(F) # 99.99937618244566

    z_index = (z>F_min) & (z<F_max)
    z = z[z_index]
    print(len(z))

    x, tilde_m_of_minus_z, m = Stieltjes_transform_Theorem3_psi_eigenvalues(z, c, sigma_star, M, psi_eigenvalues)
    print('Successfully computed m in MV')


    # Check Theorem 3 Eq. (12) with c
    Eq12_quotient_c, m_aPsi_c, Eq12_RHS_c = Check_Theorem3_Eq12(m, z, c, sigma_star, M, psi_eigenvalues)
    print('Eq12_quotient_c = %s'%(Eq12_quotient_c))

    # # Plot to check
    # plot_x_vs_tilde_m(z, x, tilde_m_of_minus_z, c)
    # plot_x_vs_m(z, x, m, c)
    # plot_mMarcenkoPastur_vs_m(z, m_MarcenkoPastur, m, c)

    # For MSRR
    F_cM = F_tildem_psi_eigenvalues(m_tilde, c_M, sigma_star, M, psi_eigenvalues)
    F_cM_min = min(F_cM)  # -0.12878060000928507
    print(F_cM_min)
    F_cM_max = max(F_cM)  # 96.88091222829783
    print(F_cM_max)
    # Plot F(\tilde{m})
    plot_F_tildem(m_tilde, F, F_cM)


    Fprime_c = np.array([fprime_function_psi_eigenvalues(m_tilde[i], c, z, sigma_star, M, psi_eigenvalues) for i in range(len(m_tilde))])

    Fprime_c_M = np.array([fprime_function_psi_eigenvalues(m_tilde[i], c_M, z, sigma_star, M, psi_eigenvalues) for i in range(len(m_tilde))])

    # fig, ax = plt.subplots()
    # # make a plot
    # l1 = plt.plot(m_tilde, Fprime_c, color='red',
    #          linestyle='solid', linewidth=3, label = 'F\', c')
    # l2 = plt.plot(m_tilde, Fprime_c_M, color='blue',
    #          linewidth=1.5, label='F\', c_M')

    z_index = (z > F_cM_min) & (z < F_cM_max)
    z = z[z_index]
    print(len(z)) # [2:len(z)]

    # # m_Psi(z) from Eq. (10), used for initialization of solve_m_Theorem3(alpha_Psi, gamma_Psi, x, -z, c)
    # x_MSRR = m_Psi_z(sigma_star, -z, M, psi_eigenvalues)
    # # x = list(m_tilde_initialization_upper_bound(sigma_star, c, psi_eigenvalues)*np.ones(len(z)))
    #
    # # tilde m solved from Theorem3
    # tilde_m_of_minus_z_MSRR = solve_m_Theorem3_psi_eigenvalues(x, c_M, -z, sigma_star, M, psi_eigenvalues)
    #
    # # m solved from Theorem3, m should be equal to x
    # q_minus_z_MSRR = map_from_tilde_m_to_m(tilde_m_of_minus_z_MSRR, -z, c_M)

    x_MSRR, tilde_m_of_minus_z_MSRR, q_minus_z_MSRR = Stieltjes_transform_Theorem3_psi_eigenvalues(z, c_M, sigma_star, M,
                                                             psi_eigenvalues)  # marcenko_pastur(1, c_M, z) # Prop 12
    # z[0:1] doesn't work
    print('Successfully computed q_minus_z in MSRR')

    # Plot to check
    # plot_x_vs_tilde_m(z, x_MSRR, tilde_m_of_minus_z_MSRR, c_M)
    plot_x_vs_m(z, x_MSRR, q_minus_z_MSRR, c_M)
    # plot_mMarcenkoPastur_vs_m(z, m_MarcenkoPastur, m, c)

    # Check Theorem 3
    Eq12_quotient_c_M, m_aPsi_c_M, Eq12_RHS_c_M = Check_Theorem3_Eq12(q_minus_z_MSRR, z, c_M, sigma_star, M, psi_eigenvalues)
    print('Eq12_quotient_c_M = %s' % (Eq12_quotient_c_M))

    # make a plot
    plot_Eq12_check(z, q_minus_z_MSRR, Eq12_RHS_c_M, m_aPsi_c_M, Eq12_quotient_c_M, c_M)

    # fig, ax = plt.subplots()
    # # make a plot
    # l1 = plt.plot(m_tilde, F_cM, color='red',
    #          linestyle='solid', linewidth=3, label = 'MSRR SRU')




