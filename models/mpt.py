import cvxpy as cp
import numpy as np

def optimize_mpt(mu, Sigma, max_w=0.30, long_only=True, cash_buffer=0.02):
    n = len(mu)
    w = cp.Variable(n)
    ret = mu @ w
    risk = cp.quad_form(w, Sigma + 1e-8*np.eye(n))
    cons = [cp.sum(w) == 1.0 - cash_buffer, w <= max_w]
    if long_only:
        cons.append(w >= 0)
    prob = cp.Problem(cp.Maximize(ret/cp.sqrt(risk)), cons)
    prob.solve(solver=cp.OSQP, verbose=False)
    val = np.array(w.value).flatten()
    if val is None:
        # fallback: equal weight
        val = np.ones(n)/n * (1.0 - cash_buffer)
    return val