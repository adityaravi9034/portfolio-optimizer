# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
import cvxpy as cp
import numpy as np

def optimize_mpt(mu, Sigma, max_w=0.30, long_only=True, cash_buffer=0.02, risk_aversion=10.0):
    """
    Mean-variance optimization (convex):
      minimize   risk_aversion * w^T Σ w  -  μ^T w
      subject to sum(w) = 1 - cash_buffer, bounds.
    """
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    n = len(mu)
    # small ridge to ensure PSD
    Sigma = Sigma + 1e-8 * np.eye(n)

    w = cp.Variable(n)
    objective = cp.Minimize(risk_aversion * cp.quad_form(w, Sigma) - mu @ w)

    cons = [cp.sum(w) == 1.0 - cash_buffer, w <= max_w]
    if long_only:
        cons.append(w >= 0)

    prob = cp.Problem(objective, cons)
    prob.solve(solver=cp.OSQP, verbose=False)

    val = np.array(w.value).flatten() if w.value is not None else None
    if val is None or not np.all(np.isfinite(val)):
        # fallback: equal-weighted within caps
        val = np.ones(n) / n * (1.0 - cash_buffer)
        val = np.minimum(val, max_w)
        val = val / val.sum() * (1.0 - cash_buffer)
    return val
