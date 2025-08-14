import argparse, yaml
from pathlib import Path
import numpy as np
import pandas as pd
from utils.io import PROC, load_df, save_df
from utils.constraints import project_weights
from utils.ticks import full_universe
from models.mpt import optimize_mpt
from models.risk_parity import risk_parity
from models.factor import factor_target_weights
from models.ml_xgb import train_xgb, preds_to_weights

TRADING_DAYS = 252

def expected_returns(mu: pd.DataFrame):
    return mu.iloc[-1].fillna(0.0)

def covariance(returns: pd.DataFrame, span=60):
    # EWMA covariance
    ret = returns.sub(returns.mean()).fillna(0)
    lam = 2/(span+1)
    S = np.cov(ret.dropna().T)
    return pd.DataFrame(S, index=returns.columns, columns=returns.columns)

def main(cfg):
    mu = load_df(PROC/'feat_mu.csv')
    vol = load_df(PROC/'feat_vol.csv')
    prices = load_df(PROC.parent.parent/'data/raw/prices.csv')
    rets = np.log(prices).diff().dropna()

    exp_mu = expected_returns(mu)
    Sigma = covariance(rets.tail(252))

    symbols = exp_mu.index.tolist()

    # --- MPT
    w_mpt = optimize_mpt(exp_mu.values, Sigma.values,
                         max_w=cfg['constraints']['max_weight'],
                         long_only=cfg['constraints']['long_only'],
                         cash_buffer=cfg['constraints']['cash_buffer'])

    # --- Risk Parity
    w_rp = risk_parity(Sigma.values, cash_buffer=cfg['constraints']['cash_buffer'])

    # --- Factor (toy target: tilt to momentum via 12m momentum ranks)
    mom12 = load_df(PROC/'feat_mom12.csv').iloc[-1].fillna(0)
    ranks = mom12.rank(pct=True)
    B = np.vstack([ranks.values])  # 1-factor proxy
    f_target = np.array([0.7])
    w_factor = factor_target_weights(B, f_target, cap=cfg['constraints']['max_weight'], cash_buffer=cfg['constraints']['cash_buffer'])

    # --- ML (XGB) simple demo: predict next 21d return with last features
    X = pd.concat([
        load_df(PROC/'feat_mu.csv'),
        load_df(PROC/'feat_vol.csv'),
        load_df(PROC/'feat_mom6.csv'),
        load_df(PROC/'feat_mom12.csv')
    ], axis=1, keys=['mu','vol','m6','m12'])

    # Build a very simple panel → per-asset vector at last date
    last = X.iloc[-1]
    cols = X.columns
    # Train per-asset with naive labels (shifted returns). For brevity, use last row only for prediction.
    preds = (last['mu'] + 0.5*last['m12']).fillna(0.0).values
    w_ml = preds_to_weights(preds, cap=cfg['constraints']['max_weight'], cash=cfg['constraints']['cash_buffer'])

    weights = pd.DataFrame({
        'MPT': w_mpt,
        'RiskParity': w_rp,
        'Factor': w_factor,
        'ML_XGB': w_ml
    }, index=symbols)

    save_df(weights, PROC/'latest_weights.csv')
    print('Saved weights to', PROC/'latest_weights.csv')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    main(cfg)