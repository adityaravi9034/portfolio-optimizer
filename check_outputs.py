# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
# check_outputs.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)

ROOT = Path(__file__).resolve().parents[0]
PROC = ROOT / "data" / "processed"
RAW  = ROOT / "data" / "raw"

FILES = [
    RAW / "prices.csv",
    RAW / "macro_fred.csv",
    PROC / "feat_mu.csv",
    PROC / "feat_vol.csv",
    PROC / "feat_mom6.csv",
    PROC / "feat_mom12.csv",
    PROC / "feat_macro.csv",
    PROC / "latest_weights.csv",
    PROC / "weights_timeseries.csv",
    PROC / "metrics_by_model.csv",
    PROC / "equity_curves.csv",
    PROC / "stress_summary.csv",
    PROC / "attribution_return.csv",
    PROC / "attribution_risk.csv",
]

def read_ts(path: Path, allow_multiindex=False):
    if not path.exists(): return None
    if allow_multiindex:
        try:
            df = pd.read_csv(path, header=[0,1], index_col=0)
            if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
                df = pd.read_csv(path, index_col=0)
        except Exception:
            df = pd.read_csv(path, index_col=0)
    else:
        df = pd.read_csv(path, index_col=0)
    # robust datetime parse
    try:
        df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
        if df.index.isna().any():
            df.index = pd.to_datetime(df.index, errors="coerce")
    except Exception:
        pass
    return df

def read_table(path: Path, allow_multiindex=False):
    if not path.exists(): return None
    if allow_multiindex:
        try:
            df = pd.read_csv(path, header=[0,1], index_col=0)
            if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
                df = pd.read_csv(path, index_col=0)
        except Exception:
            df = pd.read_csv(path, index_col=0)
    else:
        df = pd.read_csv(path, index_col=0)
    return df

def basic_summary(name, df):
    print(f"\n{name}")
    print("-"*len(name))
    print(f"shape={df.shape}  index_name={df.index.name!r}  dtypes_unique={sorted(set(map(str, df.dtypes)))}")
    # head/tail
    print("\nhead(3):")
    print(df.head(3))
    print("\ntail(3):")
    print(df.tail(3))
    # stats (numeric only)
    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        desc = num.describe().T[["mean","std","min","max"]]
        print("\nsummary stats (numeric cols):")
        print(desc.head(10))
    # NaN diagnostics
    na_cols = df.isna().mean().sort_values(ascending=False)
    worst = na_cols[na_cols > 0].head(10)
    if not worst.empty:
        print("\nNaN ratio (worst 10):")
        print((worst*100).round(2).astype(str) + "%")

def check_prices_and_features():
    prices = read_ts(RAW/"prices.csv")
    if prices is None:
        print("❌ data/raw/prices.csv missing"); return None, None
    basic_summary("prices.csv", prices)
    if not prices.index.is_monotonic_increasing:
        print("⚠️ prices index not sorted (fix by sort_index())")
    if prices.isna().all().any():
        print("⚠️ some columns are all-NaN:", prices.columns[prices.isna().all()].tolist())

    mu  = read_ts(PROC/"feat_mu.csv")
    vol = read_ts(PROC/"feat_vol.csv")
    if mu is not None and vol is not None:
        # check feature alignment with prices
        mis_mu  = sorted(set(mu.columns)  - set(prices.columns))
        mis_vol = sorted(set(vol.columns) - set(prices.columns))
        if mis_mu or mis_vol:
            print("⚠️ feature/price column mismatch:",
                  {"mu_only": mis_mu, "vol_only": mis_vol})
    return prices, mu

def check_weights_and_walkforward(prices):
    lw  = read_table(PROC/"latest_weights.csv")
    wts = read_ts(PROC/"weights_timeseries.csv", allow_multiindex=True)
    metrics = read_table(PROC/"metrics_by_model.csv")

    if lw is None:
        print("❌ latest_weights.csv missing")
    else:
        basic_summary("latest_weights.csv", lw)
        # weights per model sum check
        sums = lw.sum(axis=0)
        if (sums > 1.05).any() or (sums < 0.90).any():
            print("⚠️ column sums not close to 1.0:", sums.to_dict())
        neg = (lw < -1e-9).sum().sum()
        if neg:
            print(f"⚠️ negative weights found (count={int(neg)})")

    if wts is None:
        print("❌ weights_timeseries.csv missing")
    else:
        name = "weights_timeseries.csv"
        if isinstance(wts.columns, pd.MultiIndex) and wts.columns.nlevels == 2:
            print(f"\n{name} (MultiIndex columns: (model, asset))")
            models = sorted(set(wts.columns.get_level_values(0)))
            assets = sorted(set(wts.columns.get_level_values(1)))
            print("models:", models)
            print("assets (first 12):", assets[:12], "…")
            # check per-model weight sum sanity at last date
            last = wts.iloc[-1]
            sums = last.groupby(level=0).sum()
            off = sums[(sums < 0.90) | (sums > 1.05)]
            if not off.empty:
                print("⚠️ last-row weight sums off ~1:", off.to_dict())
        else:
            basic_summary(name, wts)

    if metrics is not None:
        basic_summary("metrics_by_model.csv", metrics)

    # equity curves
    eq = read_ts(PROC/"equity_curves.csv")
    if eq is not None:
        basic_summary("equity_curves.csv", eq)
        # sanity: last values should be >0
        if (eq.iloc[-1] <= 0).any():
            print("⚠️ nonpositive equity values at the end")

    # stress summary
    stress = read_table(PROC/"stress_summary.csv")
    if stress is not None:
        basic_summary("stress_summary.csv", stress)

def check_attribution():
    a_ret  = read_table(PROC/"attribution_return.csv", allow_multiindex=True)
    a_risk = read_table(PROC/"attribution_risk.csv", allow_multiindex=True)
    if a_ret is None and a_risk is None:
        print("❌ attribution files missing"); return
    if a_ret is not None:
        basic_summary("attribution_return.csv", a_ret)
    if a_risk is not None:
        basic_summary("attribution_risk.csv", a_risk)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-on-warn", action="store_true",
                    help="Exit with code 1 if common issues are detected.")
    args = ap.parse_args()

    had_warn = False

    print("🔍 Checking pipeline output files...")

    prices, mu = check_prices_and_features()
    if prices is None: return

    check_weights_and_walkforward(prices)
    check_attribution()

    print("\n✅ Check complete!")
    if args.fail_on_warn and had_warn:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
