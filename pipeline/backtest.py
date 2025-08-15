# pipeline/backtest.py
import argparse, yaml
from pathlib import Path
import numpy as np
import pandas as pd

from utils.io import RAW, PROC, load_df, save_df
from utils.metrics import sharpe, sortino, max_drawdown, apply_costs

TRADING_DAYS = 252

# ---- Safe defaults for trading costs ----
DEFAULT_COSTS = {
    "commission_bps": 1.0,   # 1 bp commission
    "slippage_bps":  1.0,    # 1 bp slippage
}

def _as_costs(cfg: dict) -> dict:
    src = cfg.get("costs", {}) if isinstance(cfg, dict) else {}
    if not isinstance(src, dict):
        src = {}
    out = {**DEFAULT_COSTS, **src}
    # coerce to float
    out["commission_bps"] = float(out.get("commission_bps", DEFAULT_COSTS["commission_bps"]))
    out["slippage_bps"]   = float(out.get("slippage_bps",  DEFAULT_COSTS["slippage_bps"]))
    return out

def backtest(prices: pd.DataFrame, weights_ts: pd.DataFrame, costs: dict):
    """
    Backtest a panel of weights (columns: (model, asset), index: dates when new weights apply)
    against daily prices. Applies trading costs when weights change on rebalance dates.
    Returns:
      - equity_curves (pd.DataFrame): columns per model
      - turnover (pd.DataFrame): per model turnover time series (optional use)
      - metrics (dict): per model summary stats
    """
    # daily log returns of assets
    rets = np.log(prices).diff().fillna(0.0)

    # ensure weights_ts index is datetime and sorted
    wts = weights_ts.copy()
    wts.index = pd.to_datetime(wts.index, utc=False)
    wts = wts.sort_index()

    # MultiIndex columns? expected form saved by walkforward: (model, asset)
    if not isinstance(wts.columns, pd.MultiIndex) or wts.columns.nlevels != 2:
        # fallback: treat as single model called "Model"
        wts.columns = pd.MultiIndex.from_product([["Model"], list(wts.columns)], names=["model","asset"])

    models = sorted(set(wts.columns.get_level_values(0)))
    assets = sorted(set(wts.columns.get_level_values(1)))

    # Align returns to asset list
    miss = [a for a in assets if a not in rets.columns]
    if miss:
        # add zero-return columns for any missing asset to keep shape consistent
        for a in miss:
            rets[a] = 0.0
        rets = rets.reindex(columns=sorted(rets.columns))

    # containers
    equity = {m: [] for m in models}
    turnover = {m: [] for m in models}
    last_w = {m: np.zeros(len(assets)) for m in models}

    # build a list of rebalance dates; apply period is (prev_date, this_date]
    rb_dates = list(wts.index)
    if len(rb_dates) < 2:
        raise ValueError("weights_timeseries must contain at least two rebalance dates.")

    for i in range(1, len(rb_dates)):
        start, end = rb_dates[i-1], rb_dates[i]
        ret_slice = rets.loc[start:end]  # includes end

        for m in models:
            target_w = (
                wts.loc[end, m]  # weights at 'end' for model m (Series indexed by asset)
                .reindex(assets)
                .fillna(0.0)
                .values
            )

            # trading cost factor and turnover vs previous allocation
            factor, t = apply_costs(
                last_w[m],
                target_w,
                costs["commission_bps"],
                costs["slippage_bps"],
            )
            last_w[m] = target_w

            # daily log portfolio returns over (start, end]
            # (ret_slice is daily log returns; @ target_w gives daily log port return; sum → period log return)
            period_log = float((ret_slice[assets] @ target_w).sum())
            period_mult = float(np.exp(period_log) * factor)

            equity[m].append((end, period_mult))
            turnover[m].append((end, t))

    # convert to equity curves that start at 1.0
    eq_curves = {}
    to_series = {}
    for m in models:
        ser = pd.Series({d: v for d, v in equity[m]}).sort_index()
        eq_curves[m] = ser.cumprod()
        to_series[m] = pd.Series({d: v for d, v in turnover[m]}).sort_index()

    equity_df = pd.DataFrame(eq_curves)
    turnover_df = pd.DataFrame(to_series)

    # summary metrics per model
    metrics = {}
    for m in models:
        eq = equity_df[m].dropna()
        if len(eq) >= 2:
            r = np.log(eq).diff().dropna()
            metrics[m] = {
                "CAGR": float(eq.iloc[-1] ** (TRADING_DAYS / max(1, len(eq))) - 1.0),
                "Vol": float(np.sqrt(TRADING_DAYS) * r.std(ddof=0)),
                "Sharpe": float(sharpe(r)),
                "Sortino": float(sortino(r)),
                "MaxDD": float(max_drawdown(eq)),
            }
        else:
            metrics[m] = {"CAGR": 0.0, "Vol": 0.0, "Sharpe": 0.0, "Sortino": 0.0, "MaxDD": 0.0}

    return equity_df, turnover_df, metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    costs = _as_costs(cfg)

    # Prices
    prices = load_df(RAW / "prices.csv")

    # Use month-end schedule consistently (and avoid 'M' deprecation)
    # (Not strictly necessary here, but keeps alignment similar to walkforward.)
    prices = prices.asfreq("B").ffill()

    # Weights time series from walk-forward
    wts_path = PROC / "weights_timeseries.csv"
    if not wts_path.exists():
        raise FileNotFoundError(f"{wts_path} not found. Run walkforward first.")
    weights_ts = pd.read_csv(wts_path, header=[0, 1], index_col=0)
    weights_ts.index = pd.to_datetime(weights_ts.index, utc=False)

    equity, turnover, metrics = backtest(prices, weights_ts, costs)

    # Save outputs
    save_df(equity, PROC / "backtest_equity_curve.csv")
    metrics_df = pd.DataFrame(metrics).T
    save_df(metrics_df, PROC / "latest_metrics.csv")

    print("Saved backtest metrics:", metrics)