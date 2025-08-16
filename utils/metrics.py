# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
# utils/metrics.py
from __future__ import annotations
import numpy as np
import pandas as pd

# ---------- classic helpers ----------
def sharpe(r: pd.Series, periods_per_year: float = 252.0) -> float:
    r = pd.Series(r).dropna()
    if r.empty:
        return 0.0
    mu = r.mean() * periods_per_year
    sigma = r.std(ddof=0) * np.sqrt(periods_per_year)
    return float(mu / sigma) if sigma > 0 else 0.0

def sortino(r: pd.Series, periods_per_year: float = 252.0) -> float:
    r = pd.Series(r).dropna()
    if r.empty:
        return 0.0
    downside = r[r < 0]
    dd = downside.std(ddof=0) * np.sqrt(periods_per_year)
    mu = r.mean() * periods_per_year
    return float(mu / dd) if dd > 0 else 0.0

def max_drawdown(eq: pd.Series) -> float:
    eq = pd.Series(eq).dropna()
    if eq.empty:
        return 0.0
    roll_max = eq.cummax()
    dd = eq / roll_max - 1.0
    return float(dd.min())

def apply_costs(
    w_prev: np.ndarray,
    w_new:  np.ndarray,
    commission_bps: float,
    slippage_bps:   float,
) -> tuple[float, float]:
    """
    Returns (trading_cost_factor, turnover).

    turnover = L1 change / 2
    notional cost = (commission_bps + slippage_bps) * turnover / 1e4
    factor = exp(-cost) to be multiplied with the gross period multiplier.
    """
    w_prev = np.asarray(w_prev, float)
    w_new  = np.asarray(w_new,  float)
    turnover = float(np.abs(w_new - w_prev).sum() / 2.0)
    bps = float(commission_bps) + float(slippage_bps)
    cost = bps * turnover / 1e4
    factor = float(np.exp(-cost))
    return factor, turnover

# ---------- frequency-aware equity metrics ----------
def _periods_per_year_from_index(idx: pd.Index) -> float:
    """Infer periods/year from a DatetimeIndex."""
    if not isinstance(idx, (pd.DatetimeIndex, pd.Index)) or len(idx) < 2:
        return 252.0
    try:
        freq = pd.infer_freq(idx)
    except Exception:
        freq = None

    if freq:
        f = str(freq).upper()
        if f.endswith("M") or f.endswith("ME"):
            return 12.0
        if f.startswith("W"):
            return 52.0
        if f in ("B", "D"):
            return 252.0

    # fallback by median day spacing
    try:
        days = np.median(np.diff(idx.values).astype("timedelta64[D]").astype(int))
    except Exception:
        return 252.0
    if days >= 27:
        return 12.0
    if days >= 5:
        return 52.0
    return 252.0

def _years_elapsed(idx: pd.DatetimeIndex) -> float:
    if not isinstance(idx, pd.DatetimeIndex) or len(idx) < 2:
        return 0.0
    delta_days = (idx[-1] - idx[0]).days
    return max(delta_days / 365.25, 1e-9)

def equity_metrics(eq: pd.Series) -> dict:
    """
    eq: equity curve (levels ~ start at 1.0), datetime index.
    Returns dict with CAGR, Vol, Sharpe, Sortino, MaxDD using correct scaling.
    """
    eq = pd.Series(eq).dropna()
    if not isinstance(eq.index, pd.DatetimeIndex):
        try:
            eq.index = pd.to_datetime(eq.index, errors="coerce")
            eq = eq[~eq.index.isna()]
        except Exception:
            pass

    if eq.empty or len(eq) < 2:
        return {"CAGR": 0.0, "Vol": 0.0, "Sharpe": 0.0, "Sortino": 0.0, "MaxDD": 0.0}

    ppy = _periods_per_year_from_index(eq.index)
    years = _years_elapsed(eq.index)

    # use log-returns at equity frequency
    r = np.log(eq).diff().dropna()

    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol  = float(r.std(ddof=0) * np.sqrt(ppy))
    sh   = float((r.mean() * ppy) / (r.std(ddof=0) * np.sqrt(ppy))) if r.std(ddof=0) > 0 else 0.0
    downside = r[r < 0]
    srt = float((r.mean() * ppy) / (downside.std(ddof=0) * np.sqrt(ppy))) if downside.std(ddof=0) > 0 else 0.0
    mdd  = max_drawdown(eq)

    return {"CAGR": cagr, "Vol": vol, "Sharpe": sh, "Sortino": srt, "MaxDD": mdd}

def sanity_check_metrics(metrics: dict, name: str = "") -> list[str]:
    """
    Simple guard-rails to flag suspicious stats.
    """
    warns = []
    if metrics.get("CAGR", 0) > 0.50:
        warns.append(f"{name} CAGR > 50% — check leakage/annualization.")
    if metrics.get("Vol", 0)  > 0.50:
        warns.append(f"{name} Vol > 50% — excessive risk or wrong scaling.")
    if metrics.get("Sharpe", 0) > 3.5:
        warns.append(f"{name} Sharpe > 3.5 — likely overfit or scaling issue.")
    return warns