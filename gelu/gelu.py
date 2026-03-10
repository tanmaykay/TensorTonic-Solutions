import numpy as np
import math

def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (exact version using erf).
    x: list or np.ndarray
    Return: np.ndarray of same shape (dtype=float)
    """
    x=np.asarray(x)
    erf_vec = np.vectorize(math.erf)
    return (x*(1+erf_vec(x/math.sqrt(2))))/2
    pass
