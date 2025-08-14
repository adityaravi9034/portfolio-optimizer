import numpy as np
from numpy.linalg import lstsq

# Simple factor targeting: minimize ||B w - f*|| subject to sum(w)=1-cash, w>=0, w<=cap

def factor_target_weights(B, f_target, cap=0.30, cash_buffer=0.02):
    # unconstrained least squares as a starting point
    w0, *_ = lstsq(B, f_target, rcond=None)
    w0 = np.clip(w0, 0, cap)
    s = w0.sum()
    if s > 0:
        w0 = w0 / s * (1.0 - cash_buffer)
    else:
        n = B.shape[1]
        w0 = np.ones(n)/n * (1.0 - cash_buffer)
    return w0