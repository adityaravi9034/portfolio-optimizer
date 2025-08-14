import numpy as np
import pandas as pd

TRADING_DAYS = 252

def sharpe(returns: pd.Series, rf=0.0):
    ex = returns - rf/ TRADING_DAYS
    if ex.std(ddof=0) == 0:
        return 0.0
    return np.sqrt(TRADING_DAYS) * ex.mean() / ex.std(ddof=0)

def sortino(returns: pd.Series, rf=0.0):
    downside = returns[returns < 0]
    ds = downside.std(ddof=0)
    if ds == 0:
        return 0.0
    return np.sqrt(TRADING_DAYS) * (returns.mean() - rf/TRADING_DAYS) / ds

def max_drawdown(cum: pd.Series):
    roll_max = cum.cummax()
    dd = cum/roll_max - 1.0
    return dd.min()

def apply_costs(prev_w, new_w, commission_bps=1.0, slippage_bps=2.0):
    turnover = float(np.abs(new_w - prev_w).sum())
    cost = turnover * (commission_bps + slippage_bps)/10000.0
    return max(0.0, 1.0 - cost), turnover