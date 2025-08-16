# pipeline/stress.py
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

from utils.io import RAW, PROC, load_df

STRESS_PERIODS = [
    ("GFC_2008",        "2007-10-01", "2009-03-31"),
    ("EuroCrisis_2011", "2011-05-01", "2011-10-03"),
    ("China_2015",      "2015-06-01", "2016-02-11"),
    ("Volmageddon_2018","2018-09-20", "2018-12-24"),
    ("COVID_Crash",     "2020-02-19", "2020-03-23"),
    ("Inflation_2022",  "2022-01-03", "2022-10-14"),
]

def _overlap_slice(df: pd.DataFrame | pd.Series, start: str, end: str):
    if df is None or df.empty:
        return df.iloc[0:0]
    s = pd.to_datetime(start); e = pd.to_datetime(end)
    # clamp to available range
    lo = max(df.index.min(), s)
    hi = min(df.index.max(), e)
    if lo >= hi:
        return df.iloc[0:0]
    return df.loc[lo:hi]

def _tot_return_from_level(level: pd.Series) -> float:
    if level is None or len(level) < 2:
        return np.nan
    return float(level.iloc[-1] / level.iloc[0] - 1.0)

def _max_dd(level: pd.Series) -> float:
    if level is None or len(level) < 2:
        return np.nan
    roll_max = level.cummax()
    dd = level / roll_max - 1.0
    return float(dd.min())

def run_stress():
    # Prices (asset-level)
    prices = load_df(RAW / "prices.csv")  # daily close levels
    if prices is not None and not prices.empty:
        prices = prices.asfreq("B").ffill()

    # Equity curves (portfolio-level)
    eq_path = PROC / "equity_curves.csv"
    eq = None
    if eq_path.exists() and eq_path.stat().st_size > 0:
        eq = pd.read_csv(eq_path, index_col=0, parse_dates=True)

    rows = []

    # ---- portfolio stress (per model) ----
    if isinstance(eq, pd.DataFrame) and not eq.empty:
        for nm, s, e in STRESS_PERIODS:
            win = _overlap_slice(eq, s, e)
            if win is None or win.empty:
                continue
            for model in win.columns:
                lev = win[model].dropna()
                if len(lev) < 2:
                    continue
                rows.append({
                    "period": nm,
                    "start": pd.to_datetime(s).date().isoformat(),
                    "end":   pd.to_datetime(e).date().isoformat(),
                    "level": "portfolio",
                    "name":  model,
                    "total_return": _tot_return_from_level(lev),
                    "max_drawdown": _max_dd(lev),
                })

    # ---- asset stress (per ticker) ----
    if isinstance(prices, pd.DataFrame) and not prices.empty:
        for nm, s, e in STRESS_PERIODS:
            win = _overlap_slice(prices, s, e)
            if win is None or win.empty:
                continue
            for col in win.columns:
                px = win[col].dropna()
                if len(px) < 2:
                    continue
                lev = px / px.iloc[0]  # normalize to 1 at window start
                rows.append({
                    "period": nm,
                    "start": pd.to_datetime(s).date().isoformat(),
                    "end":   pd.to_datetime(e).date().isoformat(),
                    "level": "asset",
                    "name":  col,
                    "total_return": _tot_return_from_level(lev),
                    "max_drawdown": _max_dd(lev),
                })

    out_cols = ["period","start","end","level","name","total_return","max_drawdown"]
    out = pd.DataFrame(rows, columns=out_cols)
    (PROC / "stress_periods.csv").parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROC / "stress_periods.csv", index=False)
    print(f"Saved {len(out)} stress rows -> {PROC / 'stress_periods.csv'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=False)  # kept for consistency
    _ = ap.parse_args()
    run_stress()