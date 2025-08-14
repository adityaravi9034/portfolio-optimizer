import numpy as np

def project_weights(w, long_only=True, min_w=0.0, max_w=0.3, cash_buffer=0.02):
    w = np.array(w, dtype=float)
    if long_only:
        w = np.clip(w, 0.0, None)
    w = np.minimum(w, max_w)
    total = w.sum()
    if total > 0:
        w = w / total * (1.0 - cash_buffer)
    return w