# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
# pipeline/walkforward.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

from utils.io import RAW, PROC, load_df, save_df
from utils.metrics import apply_costs, equity_metrics, sanity_check_metrics
from models.mpt import optimize_mpt
from models.risk_parity import risk_parity
from models.factor import factor_target_weights
from models.ml_xgb import preds_to_weights   # heuristic for WF; pooled ML trained in optimize.py


def _clean_weights(w: pd.Series, *, cap: float, cash: float, long_only: bool = True) -> pd.Series:
    """Clip to [0, cap], renormalize to (1 - cash). Guarantees simplex."""
    x = w.copy().astype(float).fillna(0.0)
    if long_only:
        x = x.clip(lower=0.0)
    if cap is not None:
        x = x.clip(upper=float(cap))
    s = x.sum()
    target_sum = max(0.0, 1.0 - float(cash))
    if s > 0:
        x = x * (target_sum / s)
    else:
        # if all zero, leave as all-cash
        x[:] = 0.0
    return x
# --- safe defaults (overridden by configs/default.yaml if present) ---
DEFAULT_COSTS = {
    "commission_bps": 1.0,   # 1 bp commission
    "slippage_bps":  1.0,    # 1 bp slippage
}
DEFAULT_LOOKBACKS = {
    "rolling_train_days": 756,   # ~3y
    "step_days": 21,             # monthly step (kept for compatibility)
    "rebalance_days": 21,        # monthly rebalance (compat)
    "returns_days": 252,         # trailing mean window for ER
    "vol_days": 252,             # trailing std window for vol
}

# ---------- feature + helper routines ----------
def ewma_cov(rets: pd.DataFrame, span: int = 60) -> pd.DataFrame:
    """Simple robust covariance (fallback to sample if needed)."""
    try:
        x = rets - rets.mean()
        S = np.cov(x.dropna().T)
        return pd.DataFrame(S, index=rets.columns, columns=rets.columns)
    except Exception:
        return rets.cov()

def compute_features_at(
    prices: pd.DataFrame,
    end_idx: pd.Timestamp,
    returns_days: int,
    vol_days: int,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.DataFrame]:
    """
    Features at a given end date (using ONLY info up to end_idx):
      - mu: rolling mean of log returns (per-asset)
      - vol: rolling std of log returns (per-asset)
      - mom12: 12m simple momentum
      - Sigma: EWMA/sample covariance of recent log returns
    """
    px = prices.loc[:end_idx]
    rets = np.log(px).diff().dropna()

    mu    = rets.rolling(returns_days).mean().iloc[-1].fillna(0.0)
    vol   = rets.rolling(vol_days).std().iloc[-1].fillna(0.0)
    mom12 = px.pct_change(252).iloc[-1].fillna(0.0)
    Sigma = ewma_cov(rets.tail(252))

    return mu, vol, mom12, Sigma

def gen_weights_all_models(mu, vol, mom12, Sigma, cfg) -> dict[str, pd.Series]:
    symbols = mu.index
    cons = cfg.get("constraints", {}) or {}
    cap  = float(cons.get("max_weight", 0.30))
    cash = float(cons.get("cash_buffer", 0.00))
    long_only = bool(cons.get("long_only", True))

    # MPT (raw)
    w_mpt = pd.Series(
        optimize_mpt(mu.values, Sigma.values,
                     max_w=cap, long_only=long_only,
                     cash_buffer=cash, risk_aversion=10.0),
        index=symbols
    )
    w_mpt = _clean_weights(w_mpt, cap=cap, cash=cash, long_only=long_only)

    # Risk Parity (raw)
    w_rp = pd.Series(
        risk_parity(Sigma.values, cash_buffer=cash),
        index=symbols
    )
    w_rp = _clean_weights(w_rp, cap=cap, cash=cash, long_only=long_only)

    # Factor (raw)
    ranks = mom12.rank(pct=True)
    B = np.vstack([ranks.values])
    f_target = np.array([0.7])
    w_factor = pd.Series(
        factor_target_weights(B, f_target, cap=cap, cash_buffer=cash),
        index=symbols
    )
    w_factor = _clean_weights(w_factor, cap=cap, cash=cash, long_only=long_only)

    # ML proxy (raw)
    preds = (mu + 0.5 * mom12).values
    w_ml = pd.Series(
        preds_to_weights(preds, cap=cap, cash=cash),
        index=symbols
    )
    w_ml = _clean_weights(w_ml, cap=cap, cash=cash, long_only=long_only)

    return {
        "MPT": w_mpt,
        "RiskParity": w_rp,
        "Factor": w_factor,
        "ML_XGB": w_ml,
    }
# ---------- main WF routine ----------
def walk_forward(prices: pd.DataFrame, cfg: dict):
    """
    Rolling, monthly (ME) walk-forward using month-end rebalances.
    - waits for a training window (rolling_train_days)
    - computes features at train_end (point-in-time)
    - applies target weights over the next month (apply_start, apply_end]
    - applies trading costs on each rebalance (commission + slippage)
    Returns:
      weights_ts (MultiIndex columns: (model, asset)),
      equity_curves (per model),
      metrics_df (per model),
      warnings (list[str])
    """
    # ----- lookbacks -----
    lb_src = cfg.get("lookbacks", {}) if isinstance(cfg, dict) else {}
    if not isinstance(lb_src, dict):
        lb_src = {}
    lbs = {**DEFAULT_LOOKBACKS, **lb_src}
    train_days   = int(lbs.get("rolling_train_days"))
    returns_days = int(lbs.get("returns_days"))
    vol_days     = int(lbs.get("vol_days"))

    # ----- costs -----
    cost_src = cfg.get("costs", {}) if isinstance(cfg, dict) else {}
    if not isinstance(cost_src, dict):
        cost_src = {}
    costs = {**DEFAULT_COSTS, **cost_src}
    commission_bps = float(costs.get("commission_bps", DEFAULT_COSTS["commission_bps"]))
    slippage_bps   = float(costs.get("slippage_bps",  DEFAULT_COSTS["slippage_bps"]))

    # ----- schedule -----
    monthly = prices.resample("ME").last().index  # month-end dates
    if len(monthly) == 0:
        raise ValueError("No monthly index could be derived from prices; check input prices.")

    start_idx = max(0, train_days // 21)  # approx number of months for train window
    start_idx = min(len(monthly) - 1, start_idx)
    start = monthly[start_idx]

    rebal_dates = monthly[monthly >= start]
    if len(rebal_dates) < 2:
        raise ValueError("Not enough history after training window to run walk-forward.")

    # daily log returns
    rets = np.log(prices).diff().fillna(0.0)

    models = ["MPT", "RiskParity", "Factor", "ML_XGB"]
    weights_by_model: dict[str, list[pd.Series]] = {m: [] for m in models}
    perf_by_model: dict[str, list[tuple[pd.Timestamp, float]]] = {m: [] for m in models}
    last_w: dict[str, np.ndarray] = {m: np.zeros(prices.shape[1]) for m in models}
    turnover_log: dict[str, list[tuple[pd.Timestamp, float]]] = {m: [] for m in models}

    # ----- walk-forward loop -----
    for i in range(1, len(rebal_dates)):
        train_end   = rebal_dates[i - 1]    # features computed up to here
        apply_start = rebal_dates[i - 1]
        apply_end   = rebal_dates[i]

        mu, vol, mom12, Sigma = compute_features_at(
            prices, train_end, returns_days=returns_days, vol_days=vol_days
        )
        w_dict = gen_weights_all_models(mu, vol, mom12, Sigma, cfg)

        # save weights stamped at apply_end
        for m, wser in w_dict.items():
            wser.name = apply_end
            weights_by_model[m].append(wser)

        # portfolio performance over (apply_start, apply_end]
        period_ret_log = rets.loc[apply_start:apply_end]  # daily log returns (incl end)
        for m, wser in w_dict.items():
            # apply trading costs for the change
            factor, t = apply_costs(
                last_w[m], wser.values, commission_bps, slippage_bps
            )
            last_w[m] = wser.values

            # cumulative multiplier including costs
            port_log = float((period_ret_log @ wser).sum())  # sum daily log-returns
            period_mult = float(np.exp(port_log) * factor)
            perf_by_model[m].append((apply_end, period_mult))
            turnover_log[m].append((apply_end, t))

    # ----- build outputs -----
    # (1) weights panel -> MultiIndex columns (model, asset)
    weights_panel = {}
    for m in models:
        if weights_by_model[m]:
            df_m = pd.DataFrame(weights_by_model[m])
            df_m.index = [s.name for s in weights_by_model[m]]  # dates
            df_m.index.name = "date"
            df_m = df_m.sort_index()
            weights_panel[m] = df_m
        else:
            weights_panel[m] = pd.DataFrame(index=[], columns=prices.columns)

    weights_ts = pd.concat(weights_panel, axis=1)
    weights_ts.sort_index(axis=1, inplace=True)

    # (2) equity curves (start at 1.0)
    equity_by_model: dict[str, pd.Series] = {}
    for m in models:
        if perf_by_model[m]:
            ser = pd.Series({d: v for d, v in perf_by_model[m]}).sort_index()
            equity_by_model[m] = ser.cumprod()
        else:
            equity_by_model[m] = pd.Series(dtype=float)

    equity_curves = pd.concat(equity_by_model, axis=1)

    # (3) metrics + warnings
    metrics: dict[str, dict] = {}
    warnings: list[str] = []
    for m, eq in equity_by_model.items():
        if isinstance(eq, pd.Series) and len(eq) >= 2:
            met = equity_metrics(eq)
            metrics[m] = met
            warnings += sanity_check_metrics(met, name=m)
        else:
            metrics[m] = {"CAGR": 0.0, "Vol": 0.0, "Sharpe": 0.0, "Sortino": 0.0, "MaxDD": 0.0}
    metrics_df = pd.DataFrame(metrics).T

    return weights_ts, equity_curves, metrics_df, warnings

# ---------- CLI ----------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    prices = load_df(RAW / "prices.csv")

    # ensure business-daily and forward-fill
    prices = prices.asfreq("B").ffill()

    weights_ts, equity_curves, metrics_df, warnings = walk_forward(prices, cfg)

    # save artifacts
    save_df(weights_ts,   PROC / "weights_timeseries.csv")
    save_df(equity_curves,PROC / "equity_curves.csv")
    save_df(metrics_df,   PROC / "metrics_by_model.csv")
    if warnings:
        (PROC / "run_warnings.json").write_text(json.dumps({"warnings": warnings}, indent=2))

    print("Saved walk-forward:",
          "weights_timeseries.csv, equity_curves.csv, metrics_by_model.csv")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print("  -", w)