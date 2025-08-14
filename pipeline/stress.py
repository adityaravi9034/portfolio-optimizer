# pipeline/stress.py
import argparse
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from utils.io import RAW, PROC, load_df

# Define stress windows (peak->trough style)
STRESS_WINDOWS = {
    "GFC_2008": ("2007-10-01", "2009-03-09"),
    "COVID_2020": ("2020-02-19", "2020-03-23"),
    # add more if you like:
    # "Taper_2013": ("2013-05-22", "2013-06-24"),
    # "Inflation_2022": ("2022-01-03", "2022-10-12"),
}

def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure datetime index, sorted, business-day filled forward
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=False, errors="coerce")
    df = df.sort_index().ffill()
    # drop any rows with NaT index (would show up as 1970 in CSV otherwise)
    df = df[~df.index.isna()]
    return df

def window_return(prices: pd.DataFrame, start: str, end: str) -> pd.Series:
    # Slice and compute cumulative return = last / first - 1
    px = prices.loc[start:end]
    if px.empty or len(px) < 2:
        return pd.Series(0.0, index=prices.columns)
    ret = px.iloc[-1] / px.iloc[0] - 1.0
    return ret.replace([np.inf, -np.inf], np.nan).fillna(0.0)

def portfolio_return(asset_returns: pd.Series, weights: pd.Series) -> float:
    # Align and compute w' * r
    w = weights.reindex(asset_returns.index).fillna(0.0)
    return float((w * asset_returns).sum())

def build_stress_summary(prices: pd.DataFrame, latest_weights: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prices = clean_prices(prices)

    for label, (start, end) in STRESS_WINDOWS.items():
        ar = window_return(prices, start, end)

        # asset-level rows
        for asset, r in ar.items():
            rows.append(
                {
                    "period": label,
                    "start": pd.to_datetime(start),
                    "end": pd.to_datetime(end),
                    "level": "asset",
                    "name": asset,
                    "return": r,
                }
            )

        # portfolio rows per model (static weights over the window)
        for model in latest_weights.columns:
            pr = portfolio_return(ar, latest_weights[model])
            rows.append(
                {
                    "period": label,
                    "start": pd.to_datetime(start),
                    "end": pd.to_datetime(end),
                    "level": "portfolio",
                    "name": model,
                    "return": pr,
                }
            )

    out = pd.DataFrame(rows).sort_values(["period", "level", "name"])
    return out

def main(cfg_path: str):
    # Load inputs
    prices = load_df(RAW / "prices.csv")
    lw = pd.read_csv(PROC / "latest_weights.csv", index_col=0)

    # Build and save
    summary = build_stress_summary(prices, lw)
    outp = PROC / "stress_summary.csv"
    outp.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outp, index=False)
    print("Saved stress summary to", outp)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    main(args.config)
