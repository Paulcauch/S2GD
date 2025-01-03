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

def generate_sparse_x(d,sparsity):

    sparsity = 1 - sparsity
    idx = np.arange(d)
    mask = np.random.binomial(1,sparsity,d)
    x = (-1)**idx * np.exp(-idx / 10.)
    x = x * mask

    return x 

def calculate_c_S2GD(L, mu, nu, h, m, beta):
    """
    Calculate the theoretical convergence constant c of S2GD 
    E[f(x_k)-f(x^*)] < c^k(f(x_0)-f(x^*))
    """
    term1 = ((1 - nu * h)**m) / (beta * mu * h * (1 - 2 * L * h))
    term2 = (2 * (L - mu) * h) / (1 - 2 * L * h)
    return term1 + term2


def calculate_c_S2GD_uniform(L, mu, nu, h):
    """
    Calculate the theoretical convergence constant c of S2GD for the uniform distribution
    E[f(x_k)-f(x^*)] < c^k(f(x_0)-f(x^*))
    """
    term1 = ((1 - nu * h)) / (mu * h * (1 - 2 * L * h))
    term2 = ((L - mu) * h) / (1 - 2 * L * h)
    return term1 + term2

def calculate_bound_sto_grad(step,L,sigma,mu,f_star,f_start,k):
    """
    Calculate the theoretical bound of SGD 
    E[f(x_k)-f(x^*)] < (step * L * sigma) / (2 * mu) + (1 - step*mu)**k * (f_start-f_star + (step * L * sigma) / (mu))
    """
    term1 = (step * L * sigma) / (2 * mu)
    term2 = (1 - step*mu)**k * (f_start-f_star + (step * L * sigma) / (mu))
    return term1 + term2

def get_an_approximate_sigma_sgd(pb,d,n,num):
    """
    Compute an approximate of the variance of the SGD 
    """
    sigma_squareds = []
    for _ in range (num):
        iteratebis = np.random.randn(d) 
        grad_individual = [pb.grad_i(iteratebis,i) for i in range(n)]
        grad_full = pb.grad(iteratebis)
        sigma_squared = np.mean([np.linalg.norm(grad_individual[i] - grad_full)**2 for i in range(n)])
        sigma_squareds.append(sigma_squared)
    return np.mean(sigma_squareds)