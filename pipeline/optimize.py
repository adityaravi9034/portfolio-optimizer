# pipeline/optimize.py
import argparse, yaml
from pathlib import Path
import numpy as np
import pandas as pd
from utils.io import RAW, PROC, load_df, save_df
from utils.constraints import project_weights
from utils.ticks import full_universe
from models.mpt import optimize_mpt
from models.risk_parity import risk_parity
from models.factor import factor_target_weights
from models.ml_xgb import train_ml, preds_to_weights

TRADING_DAYS = 252

def expected_returns(mu: pd.DataFrame):
    return mu.iloc[-1].fillna(0.0)

def covariance(returns: pd.DataFrame, span=60):
    """EWMA-ish covariance (simple fallback to sample)."""
    ret = returns.sub(returns.mean()).fillna(0)
    # Using sample covariance for stability on small universes
    S = np.cov(ret.dropna().T)
    return pd.DataFrame(S, index=returns.columns, columns=returns.columns)

def _stack_feature(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Stack wide [dates x assets] -> long with columns=[name]."""
    return df.stack().to_frame(name)

def main(cfg):
    # ---------- Load features & prices ----------
    mu   = load_df(PROC / 'feat_mu.csv')
    vol  = load_df(PROC / 'feat_vol.csv')
    m6   = load_df(PROC / 'feat_mom6.csv')
    m12  = load_df(PROC / 'feat_mom12.csv')
    prices = load_df(RAW / 'prices.csv')

    # Log returns for covariance
    rets = np.log(prices).diff().dropna()

    # Align symbol universe
    common_cols = list(set(mu.columns) & set(vol.columns) & set(m6.columns) & set(m12.columns) & set(prices.columns))
    mu, vol, m6, m12, prices, rets = mu[common_cols], vol[common_cols], m6[common_cols], m12[common_cols], prices[common_cols], rets[common_cols]

    # ---------- Deterministic inputs for optimizers ----------
    exp_mu = expected_returns(mu)
    Sigma = covariance(rets.tail(252))
    symbols = exp_mu.index.tolist()

    # ---------- MPT ----------
    w_mpt = optimize_mpt(
        exp_mu.values, Sigma.values,
        max_w=cfg['constraints']['max_weight'],
        long_only=cfg['constraints']['long_only'],
        cash_buffer=cfg['constraints']['cash_buffer']
    )

    # ---------- Risk Parity ----------
    w_rp = risk_parity(Sigma.values, cash_buffer=cfg['constraints']['cash_buffer'])

    # ---------- Factor (toy target: tilt to 12m momentum ranks) ----------
    mom12_last = m12.iloc[-1].fillna(0.0)
    ranks = mom12_last.rank(pct=True)
    B = np.vstack([ranks.values])  # 1-factor proxy
    f_target = np.array([0.7])
    w_factor = factor_target_weights(
        B, f_target,
        cap=cfg['constraints']['max_weight'],
        cash_buffer=cfg['constraints']['cash_buffer']
    )

    # ---------- Machine Learning (pooled cross-section; predicts fwd 21d returns) ----------
    # Build forward 21-day simple returns aligned at time t
    fwd21 = prices.pct_change(periods=-21)  # forward return from t -> t+21

    # Stack features and target into long format indexed by (date, asset)
    X_long = pd.concat([
        _stack_feature(mu,  'mu'),
        _stack_feature(vol, 'vol'),
        _stack_feature(m6,  'm6'),
        _stack_feature(m12, 'm12'),
    ], axis=1).dropna()

    y_long = fwd21.stack().to_frame('y')
    XY = X_long.join(y_long, how='inner').dropna()

    # Split: train on all dates except the last available; predict at the last date
    last_date = XY.index.get_level_values(0).max()
    mask_train = XY.index.get_level_values(0) < last_date
    X_train = XY.loc[mask_train, ['mu','vol','m6','m12']].values
    y_train = XY.loc[mask_train, 'y'].values

    # Prediction set: the last date rows (all assets available that day)
    X_pred_df = X_long.loc[last_date]
    X_pred = X_pred_df[['mu','vol','m6','m12']].values
    pred_assets = X_pred_df.index.tolist()  # asset order for preds

    # Train ML model based on config (xgboost / lightgbm / random_forest with fallback)
    ml_cfg = cfg.get('ml', {})
    model = train_ml(X_train, y_train, ml_cfg)

    # Predict, then convert to weights
    preds = model.predict(X_pred)

    # Map predictions back to the full symbol list (fill missing with 0)
    preds_series = pd.Series(preds, index=pred_assets).reindex(symbols).fillna(0.0)
    w_ml = preds_to_weights(
        preds_series.values,
        cap=cfg['constraints']['max_weight'],
        cash=cfg['constraints']['cash_buffer']
    )

    # ---------- Collect & save ----------
    weights = pd.DataFrame(
        {
            'MPT': w_mpt,
            'RiskParity': w_rp,
            'Factor': w_factor,
            'ML_XGB': w_ml,   # label kept as ML_XGB for UI consistency
        },
        index=symbols
    )

    save_df(weights, PROC / 'latest_weights.csv')
    print('Saved weights to', PROC / 'latest_weights.csv')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    main(cfg)
