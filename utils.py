import numpy as np
from scipy.linalg import norm 
from numpy.random import multivariate_normal, randn # Probability distributions on vectors
from scipy.linalg import toeplitz

def generate_pb_parameters(x, n, d, target_kappa, lbda, std=1e-5, corr=0.5):
    """
    Adjust the condition number of a given matrix A exactly, considering fixed regularization.
    
    Parameters
    ----------
    x : np.ndarray, shape=(d,)
        Ground truth coefficients.
    n : int
        Sample size.
    d : int
        Number of features.
    target_kappa : float
        Desired condition number including regularization.
    lbda : float
        Regularization parameter.
    std : float, default=1.
        Standard deviation of noise.
    corr : float, default=0.5
        Correlation of the feature matrix.
    
    Returns
    -------
    A_adjusted : np.ndarray, shape=(n, d)
        Matrix with adjusted condition number.
    y : np.ndarray, shape=(n,)
        Response vector.
    """

    # Generate initial matrix with correlation
    cov = toeplitz(corr ** np.arange(0, d))
    A = multivariate_normal(np.zeros(d), cov, size=n)

    # Compute desired ratio of singular values
    sigma_min = 1
    sigma_max = target_kappa*sigma_min - n/2 * lbda * (1-target_kappa)

    # Adjust singular values to match the condition number
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s_adjusted = np.linspace(sigma_min ** 0.5, sigma_max ** 0.5, len(s))
    A_adjusted = U @ np.diag(s_adjusted) @ Vt

    # Generate response vector
    noise = std * randn(n)
    y = A_adjusted.dot(x) + noise

    return A_adjusted, y