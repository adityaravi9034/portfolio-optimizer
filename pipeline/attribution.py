# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
import argparse, yaml
from pathlib import Path
import numpy as np, pandas as pd
from utils.io import RAW, PROC, load_df, save_df

def risk_contrib(w, Sigma):
    w = np.asarray(w, dtype=float)
    Sigma = np.asarray(Sigma, dtype=float)
    port_var = w @ Sigma @ w
    if port_var <= 1e-12:
        return np.zeros_like(w)
    mrc = Sigma @ w
    rc = w * mrc
    return rc / port_var

def main(cfg):
    prices = load_df(RAW/"prices.csv")
    rets = np.log(prices).diff().dropna()
    latest_w = pd.read_csv(PROC/"latest_weights.csv", index_col=0)

    win = rets.tail(252)
    mu = win.mean()
    Sigma = win.cov().values

    ret_frames = []
    risk_frames = []
    for model in latest_w.columns:
        w = latest_w[model].reindex(mu.index).fillna(0.0).values
        # return contribution (annualized)
        contrib_ret = (w * mu).rename(model) * 252
        ret_frames.append(contrib_ret.to_frame())

        # risk contribution (share of variance)
        rc = pd.Series(risk_contrib(w, Sigma), index=mu.index, name=model)
        risk_frames.append(rc)

    ret_df = pd.concat(ret_frames, axis=1)
    risk_df = pd.concat(risk_frames, axis=1)

    save_df(ret_df, PROC/"attribution_return.csv")
    save_df(risk_df, PROC/"attribution_risk.csv")
    print("Saved attribution_return.csv and attribution_risk.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    main(cfg)
