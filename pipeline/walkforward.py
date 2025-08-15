# pipeline/walkforward.py
import argparse, yaml
from pathlib import Path
import numpy as np
import pandas as pd

from utils.io import RAW, PROC, load_df, save_df
from utils.metrics import sharpe, sortino, max_drawdown, apply_costs
from models.mpt import optimize_mpt
from models.risk_parity import risk_parity
from models.factor import factor_target_weights
from models.ml_xgb import preds_to_weights   # ML here uses simple heuristic; optimize.py trains pooled ML

TRADING_DAYS = 252

# --- defaults to avoid KeyError when not present in config ---
DEFAULT_COSTS = {
    "commission_bps": 1.0,   # 1 bp commission
    "slippage_bps":  1.0,    # 1 bp slippage
}
DEFAULT_LOOKBACKS = {
    "rolling_train_days": 756,   # ~3y
    "step_days": 21,             # monthly step
    "rebalance_days": 21,        # monthly rebalance
    "returns_days": 252,         # trailing mean window for ER
    "vol_days": 252,             # trailing std window for vol
}


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
    Compute features at a given end date:
    - mu: rolling mean of log returns (per-asset)
    - vol: rolling std of log returns (per-asset)
    - mom12: 12m simple momentum
    - Sigma: EWMA/sample covariance of recent log returns
    """
    px = prices.loc[:end_idx]
    rets = np.log(px).diff().dropna()
    # trailing windows with safe defaults
    mu = rets.rolling(returns_days).mean().iloc[-1].fillna(0.0)
    vol = rets.rolling(vol_days).std().iloc[-1].fillna(0.0)
    mom12 = px.pct_change(252).iloc[-1].fillna(0.0)
    Sigma = ewma_cov(rets.tail(252))
    return mu, vol, mom12, Sigma


def gen_weights_all_models(
    mu: pd.Series,
    vol: pd.Series,
    mom12: pd.Series,
    Sigma: pd.DataFrame,
    cfg: dict,
) -> dict[str, pd.Series]:
    symbols = mu.index

    # MPT
    w_mpt = optimize_mpt(
        mu.values,
        Sigma.values,
        max_w=cfg["constraints"]["max_weight"],
        long_only=cfg["constraints"]["long_only"],
        cash_buffer=cfg["constraints"]["cash_buffer"],
        risk_aversion=10.0,
    )

    # Risk Parity
    w_rp = risk_parity(Sigma.values, cash_buffer=cfg["constraints"]["cash_buffer"])

    # Factor (momentum tilt via ranks)
    ranks = mom12.rank(pct=True)
    B = np.vstack([ranks.values])  # 1-factor proxy
    f_target = np.array([0.7])
    w_factor = factor_target_weights(
        B,
        f_target,
        cap=cfg["constraints"]["max_weight"],
        cash_buffer=cfg["constraints"]["cash_buffer"],
    )

    # ML proxy here (optimize.py has the pooled ML; for WF we keep a simple score)
    preds = (mu + 0.5 * mom12).values
    w_ml = preds_to_weights(
        preds,
        cap=cfg["constraints"]["max_weight"],
        cash=cfg["constraints"]["cash_buffer"],
    )

    return {
        "MPT": pd.Series(w_mpt, index=symbols),
        "RiskParity": pd.Series(w_rp, index=symbols),
        "Factor": pd.Series(w_factor, index=symbols),
        "ML_XGB": pd.Series(w_ml, index=symbols),
    }


def walk_forward(prices: pd.DataFrame, cfg: dict):
    """
    Rolling, monthly (ME) walk-forward backtest that:
      - waits for a training window (rolling_train_days)
      - rebalances on month-end steps
      - logs weights and period multipliers per model
    """

    # -------- Lookbacks (safe with defaults) --------
    lb_src = cfg.get("lookbacks", {}) if isinstance(cfg, dict) else {}
    if not isinstance(lb_src, dict):
        lb_src = {}
    lbs = {**DEFAULT_LOOKBACKS, **lb_src}

    train_days     = int(lbs.get("rolling_train_days"))
    step_days      = int(lbs.get("step_days"))
    rebalance_days = int(lbs.get("rebalance_days"))
    returns_days   = int(lbs.get("returns_days"))
    vol_days       = int(lbs.get("vol_days"))

    # -------- Costs (safe with defaults) --------
    cost_src = cfg.get("costs", {}) if isinstance(cfg, dict) else {}
    if not isinstance(cost_src, dict):
        cost_src = {}
    costs = {**DEFAULT_COSTS, **cost_src}
    commission_bps = float(costs.get("commission_bps", DEFAULT_COSTS["commission_bps"]))
    slippage_bps   = float(costs.get("slippage_bps",  DEFAULT_COSTS["slippage_bps"]))

    # -------- Monthly schedule --------
    monthly = prices.resample("ME").last().index  # month-end index
    if len(monthly) == 0:
        raise ValueError("No monthly index could be derived from prices; check input data.")

    # Require training history before first rebalance (~train_days)
    start_idx = max(0, train_days // 21)  # approx months
    start_idx = min(len(monthly) - 1, start_idx)
    start = monthly[start_idx]

    rebal_dates = monthly[monthly >= start]
    if len(rebal_dates) < 2:
        raise ValueError("Not enough history after training window to run walk-forward.")

    rets = np.log(prices).diff().fillna(0.0)

    # -------- Containers --------
    models = ["MPT", "RiskParity", "Factor", "ML_XGB"]
    weights_by_model: dict[str, list[pd.Series]] = {m: [] for m in models}
    perf_by_model: dict[str, list[tuple[pd.Timestamp, float]]] = {m: [] for m in models}
    last_w: dict[str, np.ndarray] = {m: np.zeros(prices.shape[1]) for m in models}
    turnover_log: dict[str, list[tuple[pd.Timestamp, float]]] = {m: [] for m in models}

    # -------- Walk-forward loop --------
    for i in range(1, len(rebal_dates)):
        train_end   = rebal_dates[i - 1]
        apply_start = rebal_dates[i - 1]
        apply_end   = rebal_dates[i]

        mu, vol, mom12, Sigma = compute_features_at(
            prices, train_end, returns_days=returns_days, vol_days=vol_days
        )
        w_dict = gen_weights_all_models(mu, vol, mom12, Sigma, cfg)

        # Store weights (timestamp = apply_end)
        for m, wser in w_dict.items():
            wser.name = apply_end
            weights_by_model[m].append(wser)

        # Portfolio performance over (apply_start, apply_end]
        period_ret_log = rets.loc[apply_start:apply_end]  # daily log returns (incl end)
        for m, wser in w_dict.items():
            # costs based on change from last weights to new target
            factor, t = apply_costs(
                last_w[m],
                wser.values,
                commission_bps,
                slippage_bps,
            )
            last_w[m] = wser.values

            # cumulative multiplier over the period with costs
            port_log = float((period_ret_log @ wser).sum())  # sum daily log-returns
            period_mult = float(np.exp(port_log) * factor)   # include trading costs
            perf_by_model[m].append((apply_end, period_mult))
            turnover_log[m].append((apply_end, t))

    # ---------- Build outputs ----------
    # 1) Weights timeseries: MultiIndex columns (model, asset)
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

    weights_ts = pd.concat(weights_panel, axis=1)  # columns: (model, asset)
    weights_ts.sort_index(axis=1, inplace=True)
    save_df(weights_ts, PROC / "weights_timeseries.csv")

    # 2) Equity curves (start at 1.0), columns per model
    equity_by_model: dict[str, pd.Series] = {}
    for m in models:
        if perf_by_model[m]:
            ser = pd.Series({d: v for d, v in perf_by_model[m]}).sort_index()
            eq = ser.cumprod()
            equity_by_model[m] = eq
        else:
            equity_by_model[m] = pd.Series(dtype=float)

    equity_curves = pd.concat(equity_by_model, axis=1)
    save_df(equity_curves, PROC / "equity_curves.csv")

    # 3) Per-model metrics
    metrics = {}
    for m, eq in equity_by_model.items():
        if len(eq) >= 2:
            r = np.log(eq).diff().dropna()
            metrics[m] = {
                "CAGR": float(eq.iloc[-1] ** (TRADING_DAYS / len(eq)) - 1.0),
                "Vol": float(np.sqrt(TRADING_DAYS) * r.std(ddof=0)),
                "Sharpe": float(sharpe(r)),
                "Sortino": float(sortino(r)),
                "MaxDD": float(max_drawdown(eq)),
            }
        else:
            metrics[m] = {"CAGR": 0.0, "Vol": 0.0, "Sharpe": 0.0, "Sortino": 0.0, "MaxDD": 0.0}

    metrics_df = pd.DataFrame(metrics).T
    save_df(metrics_df, PROC / "metrics_by_model.csv")

    print("Saved walk-forward: weights_timeseries.csv (MultiIndex), equity_curves.csv, metrics_by_model.csv")
    return weights_ts, equity_by_model, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    prices = load_df(RAW / "prices.csv")
    walk_forward(prices, cfg)