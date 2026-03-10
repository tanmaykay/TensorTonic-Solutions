import numpy as np
def sigmoid(x):
    x=np.asarray(x)
    return 1/(1+np.exp(-x))
def swish(x):
    """
    Implement Swish activation function.
    """
    x=np.asarray(x)
    return x*sigmoid(x)