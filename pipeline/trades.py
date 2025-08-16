# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
# pipeline/trades.py
import argparse
import yaml
from pathlib import Path
import pandas as pd
from utils.io import PROC, RAW, load_df

def make_trades(
    latest_w: pd.DataFrame,
    prev_w: pd.DataFrame,
    prices: pd.DataFrame,
    notional: float,
    threshold_bps: float = 10.0,
) -> dict[str, pd.DataFrame]:
    """
    Build per-model trade blotters from delta weights.
    - latest_w: DataFrame [assets x models] with current target weights
    - prev_w:   DataFrame [assets x models] with prior target weights (or zeros)
    - prices:   DataFrame [dates x assets] with latest close prices
    - notional: portfolio value used to translate weight deltas to dollars
    - threshold_bps: ignore trades where |Δw| < threshold (in basis points)
    Returns a dict: {model -> trade DataFrame}
    """
    # align symbols
    symbols = latest_w.index.intersection(prev_w.index)
    latest_w = latest_w.loc[symbols]
    prev_w = prev_w.loc[symbols]

    # last prices (forward-fill if needed)
    px_row = prices.iloc[-1].reindex(symbols).ffill()

    trades_by_model: dict[str, pd.DataFrame] = {}
    thresh = threshold_bps / 10_000.0  # to decimal

    for model in latest_w.columns:
        prev_col = prev_w[model] if model in prev_w.columns else 0.0
        delta = (latest_w[model] - prev_col).rename("delta_weight")

        # apply threshold (zero out tiny trades)
        delta = delta.where(delta.abs() >= thresh, other=0.0)

        # translate to dollars / shares
        dollars = delta * notional
        # avoid division by zero
        safe_px = px_row.replace(0.0, pd.NA)
        shares = dollars.divide(safe_px)

        blotter = pd.DataFrame(
            {
                "asset": symbols,
                "delta_weight": delta.values,
                "dollars": dollars.values,
                "last_price": px_row.values,
                "approx_shares": shares.values,
            }
        ).sort_values("delta_weight")

        trades_by_model[model] = blotter

    return trades_by_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--notional", type=float, default=100_000.0)
    ap.add_argument(
        "--threshold-bps",
        type=float,
        default=10.0,
        help="Ignore trades with |Δweight| below this threshold (in basis points). Default: 10 bps",
    )
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    # Load inputs
    latest = load_df(PROC / "latest_weights.csv")
    prices = load_df(RAW / "prices.csv")

    prev_path = PROC / "prev_weights.csv"
    if prev_path.exists():
        prev = pd.read_csv(prev_path, index_col=0)
    else:
        # first run: zero previous weights (same shape as latest)
        prev = latest.copy() * 0.0

    # Build trade blotters per model
    trades = make_trades(
        latest_w=latest,
        prev_w=prev,
        prices=prices,
        notional=args.notional,
        threshold_bps=args.threshold_bps,
    )

    # Save per-model CSVs
    outdir = PROC / "trades"
    outdir.mkdir(parents=True, exist_ok=True)
    for model, blotter in trades.items():
        (outdir / f"trades_{model}.csv").write_text(blotter.to_csv(index=False))

    # Persist current weights as "previous" for the next run
    latest.to_csv(prev_path)

    print(f"Saved trade blotters to {outdir} (threshold={args.threshold_bps} bps, notional={args.notional:,.0f})")
