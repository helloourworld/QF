import numpy as np
import multiprocessing as mp
import time
import pandas as pd
from itertools import product
from sklearn.linear_model import Ridge
import os
from sklearn.metrics import r2_score
from itertools import combinations # permutations
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
import random
from scipy import stats
import multiprocessing as mp
import itertools
from scipy.optimize import minimize
from random import seed
from random import random
from auxilliary_functions import matrix_square_root

# seed random number generator
seed(1)

# Parameter setting
list_of_z = np.array([1e-16, 1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, \
                      1e-2, 1e-1, 1, 1e+1, 1e+2, 1e+3, 1e+4, 1e+5, 1e+6]) #,\
                      # 1e+8, 1e+10, 1e+12, 1e+14]) #, 1e16, 1e18, 1e20])

N = 1
T_TRAIN = 45 # 95  # fixed
T_OOS = 50  # 12 * 10
STARTING_POINT = 500  # Don't start from 0, because 0 is nan
DEGREE = 2

number_of_combinations = 100
performance_names = ['R-square', 'SRU', 'Return', 'Risk', 'MSE']
parallel = False
check_distribution = False
demean_factors = True

if 'kz272' in os.path.expanduser('~'):
    code_path = '/gpfs/loomis/home.grace/kz272/scratch60/MSRR/Code/Sims20210615_Python/Grace_MSRR_Output'
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


def get_insample_psi_star1(signals):
    '''
    This function computes the in-sample psi_{*,1}
    :param signals:
    :return:
    '''
    M = signals.shape[0]
    T_ = signals.shape[1]
    a_matrix = np.matmul(signals, signals.T) / T_  # M \times M
    psi_star1 = 1/M * np.trace(a_matrix)
    return psi_star1, a_matrix

def get_b_star_from_TrueRsquared(rho, psi_star1):
    '''
    This function computes b_star by TrueRsquared in Eq. (45):
    True_R_squared = 1 - N/(N + np.matmul(beta.T, np.matmul(Psi, beta))) = 1 - N / (N + np.trace(sigma_hat) * b_star * psi_star1)
    Since N = 1, np.trace(sigma_hat) = 1, suppose \rho = TrueRsquared, then we have
    b_star = rho/((1 - rho)*psi_star1)
    :param rho: TrueRsquared
    :param psi_star1: get_insample_psi_star1(M, signals)
    :return:
    '''
    b_star = rho/((1 - rho)*psi_star1)
    return b_star

def get_z_star(beta, b_star, list_of_z):
    '''
    This function picks z_star by ||\beta(z_star)\\^2 = b_star
    :param beta:
    :param b_star:
    :param list_of_z:
    :return:
    '''
    beta_norm = np.sum(beta**2, 0)
    beta_error = np.array([np.abs(beta_norm - b_star_i) for b_star_i in b_star]) # len(b_star) \times len(list_of_z)
    z_star_idx = np.argmin(beta_error, 1)
    z_star = list_of_z[z_star_idx]
    return z_star, z_star_idx


def Simulate_OOS_Signals_and_Returns(in_sample_signals, in_sample_estimated_betas, T_OOS, rho):
    '''

    :param in_sample_signals: number_of_base_signals \times T
    :param in_sample_estimated_betas: number_of_base_signals \times len(list_of_z)
    :param T_OOS:
    :return: (T, M), (T, len(rho))
    '''
    in_sample_signals = in_sample_signals.T
    M = in_sample_signals.shape[0]

    # Step 1) compute psi_{*,1} in-sample = (M^{-1})trace(A_T)
    psi_star1, Psi = get_insample_psi_star1(in_sample_signals)

    # Step 2: pick a target Rsquared from Bryan's range. I suggest 1, 2, 3, 4 %
    # In simulations please indicate these value; and we will run separate simulations for each. Let us call this variable \rho
    # rho = np.array([0.01, 0.02, 0.03, 0.04])

    # Solve (45) to get b_star = \rho/ (\psi_{*,1} (1-\rho))
    b_star_array = get_b_star_from_TrueRsquared(rho, psi_star1)

    # Step 3) compute the different beta(z) for different values of z; pick the z_* for which \|\beta(z_*)\|^2 = b_star
    # We will call it \beta_star = \beta(z_*) = in-sample estimated beta that give the right r-squared

    z_star_array, z_star_idx = get_z_star(in_sample_estimated_betas, b_star_array, list_of_z)
    beta_star = [in_sample_estimated_betas[:, z_star_idx_] for z_star_idx_ in z_star_idx]

    # Step 4) Use A_T in sample as our model for \Psi
    # Step 5) simulate OOS Signals = \Psi^{1/2} * vectorOfGaussian(0,1)_lengthM
    Simulate_OOS_Signals = np.matmul(matrix_square_root(Psi), np.random.normal(0, 1, [M,T_OOS]))  # M \times 1

    # Step 6) Simulate OOS returns = OOSsimulatedSignals * beta_star + Gaussian(0,1)_length_T_OOS
    Simulate_OOS_Returns = np.array([np.matmul(Simulate_OOS_Signals.T, beta_star_) + np.random.normal(0, 1, T_OOS) for beta_star_ in beta_star])

    # Now you can run all your OOS performance functions separately on real data and on simulated data
    # E.g., you will get OOS_MSE(z; real OOS data) and OOS_MSE(z; simulated OOS data)
    # And we would like to compare these two

    return Simulate_OOS_Signals.T, Simulate_OOS_Returns.T, z_star_idx

