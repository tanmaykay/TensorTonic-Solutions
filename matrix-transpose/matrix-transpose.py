import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    A=np.asarray(A)
    ROWS=len(A[0])
    COLS=len(A)
    res=np.zeros((ROWS,COLS))
    for R in range(ROWS):
        for C in range(COLS):
            res[R][C]=A[C][R]
    return res
    
    pass
