import numpy as np

def risk_parity(Sigma, tol=1e-6, iters=5000, cash_buffer=0.02):
    n = Sigma.shape[0]
    w = np.ones(n)/n * (1.0 - cash_buffer)
    for _ in range(iters):
        mrc = Sigma @ w
        rc = w * mrc
        target = rc.mean()
        grad = rc - target
        w -= 0.01 * grad
        w = np.clip(w, 0, None)
        s = w.sum()
        if s > 0:
            w = w / s * (1.0 - cash_buffer)
        if np.linalg.norm(grad) < tol:
            break
    return w