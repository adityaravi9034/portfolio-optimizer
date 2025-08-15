import argparse, yaml
from pathlib import Path
import numpy as np, pandas as pd
from utils.io import RAW, PROC, load_df, save_df

def main(cfg):
    prices = load_df(RAW/"prices.csv")
    fred   = load_df(RAW/"macro_fred.csv")

    # Basic features on prices
    rets = np.log(prices).diff()
    mu = (rets.rolling(60).mean() * 252).dropna()
    vol = (rets.rolling(60).std(ddof=0) * np.sqrt(252)).dropna()
    mom_6m = prices.pct_change(126)
    mom_12m = prices.pct_change(252)

    save_df(mu,       PROC/"feat_mu.csv")
    save_df(vol,      PROC/"feat_vol.csv")
    save_df(mom_6m,   PROC/"feat_mom6.csv")
    save_df(mom_12m,  PROC/"feat_mom12.csv")

    # ---- Macro engineered features ----
    m = fred.copy()

    # Yield curve slope: 10y - 3m, 10y - 2y
    if {"DGS10","DGS3MO"}.issubset(m.columns):
        m["YC_10y_3m"] = m["DGS10"] - m["DGS3MO"]
    if {"DGS10","DGS2"}.issubset(m.columns):
        m["YC_10y_2y"] = m["DGS10"] - m["DGS2"]

    # Credit proxy: LQD – TLT monthly changes (levels are rates; here we use % change)
    # If both tickers in prices, we can compute credit spread on prices too — keep macro-only here.

    # VIX daily change (if present)
    if "VIX" in m.columns:
        m["VIX_chg"] = m["VIX"].pct_change()

    # CPI/GDP/PCE YoY; UNRATE level; FedFunds level
    for col in ["CPI","GDP","PCE"]:
        if col in m.columns:
            m[f"{col}_YoY"] = m[col].pct_change(12)
    # LFPR change
    if "LFPR" in m.columns:
        m["LFPR_chg"] = m["LFPR"].pct_change()

    macro_feat = m.dropna(how="all")
    macro_feat = macro_feat.reindex(prices.index, method="ffill")

    save_df(macro_feat, PROC/"feat_macro.csv")
    print("Features saved:", ["feat_mu.csv","feat_vol.csv","feat_mom6.csv","feat_mom12.csv","feat_macro.csv"])

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    main(cfg)
