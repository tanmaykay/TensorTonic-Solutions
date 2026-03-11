import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    N=len(seqs)
    if N==0:
        return np.full((0,0),pad_value)
    if max_len is None:
        L=max(len(seq) for seq in seqs)
    else:
        L=max_len
    res=np.full((N,L),pad_value)
    for i,seq in enumerate(seqs):
        trunc=seq[:L]
        res[i,:len(trunc)]=trunc
    return res