import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    x=np.asarray(x)
    if rng:
        random=rng.random(x.shape)
    else:
        random=np.random.random(x.shape)

    mask=random<(1-p)
    pattern = mask.astype(float)/(1-p)
    output=x*pattern
    return (output,pattern)
    pass