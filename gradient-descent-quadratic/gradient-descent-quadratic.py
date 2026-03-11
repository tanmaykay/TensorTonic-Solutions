import math
import numpy as np
def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    x=x0
    for _ in range(steps):
        dw=2*a*x+b
        x=x-lr*dw
    return float(x)
    pass
    