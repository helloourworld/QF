import numpy as np
from auxilliary_functions import matrix_square_root

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

def generate_Sigma_Psi_with_empirical_eigenvalues(N, a, EIGENVALUES):
    """
    This function generates Sigma or Psi with empirical eigenvalues
    For Sigma (covariance of signals across assets) generation, Sigma = (N, a_Sigma, SIGMA_EIGENVALUES)
    For Psi (covariance across signals) generation, Psi = (M, a_Psi, PSI_EIGENVALUES)
    :param N: Number of assets (N) or number of signals (M).
    :param a: herfindal index, which controls concentration. A large a corresponds to very high concentration of eigenvalues. a > 0
    :param EIGENVALUES: Signal noise variance computed from empirical analysis. sigma_squared > 0.
    :return:
    """
    eig = np.abs(EIGENVALUES[-N:])
    # # sometimes the minimal eigenvalue is already large.
    # # we need to correct it to avoid crazy numbers.
    # if eig[0] > 0.001:
    #     eig *= 0.001 / eig[0]
    eigenvalues = np.power(eig, a)
    Sigma_or_Psi = np.diag(eigenvalues)
    return Sigma_or_Psi, eigenvalues

def generate_Sigma_with_empirical_eigenvalues(N, a, Sigma_eigenvalues_multi = 1):
    """
    This function generates Sigma with empirical eigenvalues
    For Sigma (covariance of signals across assets) generation, Sigma = (N, a_Sigma, SIGMA_EIGENVALUES)
    :param N: Number of assets
    :param a: herfindal index, which controls concentration. A large a corresponds to very high concentration of eigenvalues. a > 0
    :return:
    """
    # Empirical eigenvalues
    SIGMA_EIGENVALUES = np.load('./plots_of_eigenvalues/eigenvalues_across_stocks_1994-01-31T00:00:00.000000000.npy',
                                                             allow_pickle=True)

    eig = np.abs(SIGMA_EIGENVALUES[-N:])
    # # sometimes the minimal eigenvalue is already large.
    # # we need to correct it to avoid crazy numbers.
    # if eig[0] > 0.001:
    #     eig *= 0.001 / eig[0]
    sigma_eigenvalues = np.power(eig, a)
    sigma_eigenvalues = N * sigma_eigenvalues / np.sum(sigma_eigenvalues) * Sigma_eigenvalues_multi
    Sigma = np.diag(sigma_eigenvalues)
    return Sigma, sigma_eigenvalues


def generate_Psi_with_empirical_eigenvalues(M, a, target_psi_star1 = 1):
    """
    This function generates Psi with empirical eigenvalues
    :param M: number of signals.
    :param a: herfindal index, which controls concentration. A large a corresponds to very high concentration of eigenvalues. a > 0
    :param EIGENVALUES: Signal noise variance computed from empirical analysis. sigma_squared > 0.
    :return:
    """
    # Empirical eigenvalues
    PSI_EIGENVALUES = np.load('./plots_of_eigenvalues/eigenvalues_across_signals_1994-01-31T00:00:00.000000000.npy',
                              allow_pickle=True)

    eig = np.abs(PSI_EIGENVALUES[-M:])
    # # sometimes the minimal eigenvalue is already large.
    # # we need to correct it to avoid crazy numbers.
    # if eig[0] > 0.001:
    #     eig *= 0.001 / eig[0]
    psi_eigenvalues  = np.power(eig, a)
    psi_eigenvalues = M * psi_eigenvalues / np.sum(psi_eigenvalues) * target_psi_star1
    Psi = np.diag(psi_eigenvalues)
    return Psi, psi_eigenvalues

def generate_Sigma_epsilon(N, a_Sigma_eps):
    """
    This function generates Sigma_epsilon randomly with SIGMA_EPSILON_EIGENVALUES
    :param N: Number of assets
    :param a_Sigma_eps: herfindal index, which controls concentration. A large a corresponds to very high concentration of eigenvalues.
                        When a_Sigma_eps=0, we are back to the old case when Sigma_eps=I
    :param SIGMA_EPSILON_EIGENVALUES: eigenvalues of Sigma_epsilon from empirical covariance of returns
    :return: Sigma_epsilon
    """

    SIGMA_EPSILON_EIGENVALUES = np.load('./plots_of_eigenvalues_Sigma_epsilon/eigenvalues_Sigma_epsilon_1994-01-31T00:00:00.000000000.npy',
                                   allow_pickle=True)

    # Simulate Gaussian i.i.d. Z (N*N)
    Z = np.random.random((N, N))
    values, vectors = np.linalg.eigh(Z + Z.T)
    U = vectors

    eig = np.abs(SIGMA_EPSILON_EIGENVALUES[-N:])
    eigenvalues = np.power(eig, a_Sigma_eps)

    Sigma_eps = np.matmul(np.matmul(U, np.eye(N) * eigenvalues), U.T)
    return Sigma_eps

def estimate_Sigma_epsilon(in_sample_returns, w):
    """
    This function estimates Sigma_epsilon from in_sample_returns
    :param in_sample_returns:
    :param w: penalization in Sigma_epsilon
    :return:
    """
    N, T = in_sample_returns.shape
    return np.matmul(in_sample_returns, in_sample_returns.T) / T + w * np.eye(N)


def sigma_hat_function(sigma, sigma_eps):
    """
    This is the \hat\Sigma matrix from the paper, Eq. (4). When Sigma_epsilon is not I, \hat\Sigma is not \Sigma
    :param sigma: true Sigma
    :param sigma_eps: estimated Sigma_epsilon
    :return:
    """
    tmp1 = matrix_square_root(sigma)
    tmp2 = np.linalg.inv(sigma_eps)
    sigma_hat = np.matmul(tmp1, np.matmul(tmp2, tmp1))
    eigenvalues, _ = np.linalg.eigh(sigma_hat)
    return sigma_hat, eigenvalues
