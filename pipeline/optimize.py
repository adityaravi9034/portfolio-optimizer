# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
# pipeline/optimize.py
import argparse, yaml
from pathlib import Path
import numpy as np
import pandas as pd

from utils.io import RAW, PROC, load_df, save_df
from models.mpt import optimize_mpt
from models.risk_parity import risk_parity
from models.factor import factor_target_weights
from models.ml_models import train_single_model, preds_to_weights

TRADING_DAYS = 252

def expected_returns(mu: pd.DataFrame):
    return mu.iloc[-1].fillna(0.0)

def covariance(returns: pd.DataFrame):
    ret = returns.sub(returns.mean()).fillna(0)
    S = np.cov(ret.dropna().T)
    return pd.DataFrame(S, index=returns.columns, columns=returns.columns)

def _stack_feature(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df.stack().to_frame(name)

def _safe_features() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mu   = load_df(PROC / 'feat_mu.csv')
    vol  = load_df(PROC / 'feat_vol.csv')
    m6   = load_df(PROC / 'feat_mom6.csv')
    m12  = load_df(PROC / 'feat_mom12.csv')
    prices = load_df(RAW / 'prices.csv')
    if any(x is None or x.empty for x in [mu, vol, m6, m12, prices]):
        raise RuntimeError("Missing features or prices. Run Features stage first.")
    return mu, vol, m6, m12, prices

def main(cfg):
    # ---------- Load features & prices ----------
    mu, vol, m6, m12, prices = _safe_features()

    rets = np.log(prices).diff().dropna()

    # Align universe
    common = list(set(mu.columns) & set(vol.columns) & set(m6.columns) & set(m12.columns) & set(prices.columns))
    mu, vol, m6, m12, prices, rets = mu[common], vol[common], m6[common], m12[common], prices[common], rets[common]

    # ---------- Deterministic inputs ----------
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

    # ---------- Factor (momentum tilt) ----------
    mom12_last = m12.iloc[-1].fillna(0.0)
    ranks = mom12_last.rank(pct=True)
    B = np.vstack([ranks.values])
    f_target = np.array([float(cfg.get("factor", {}).get("momentum_weight", 0.7))])
    w_factor = factor_target_weights(
        B, f_target,
        cap=cfg['constraints']['max_weight'],
        cash_buffer=cfg['constraints']['cash_buffer']
    )

    # ---------- ML (single or ensemble) ----------
    ml_weights = None
    try:
        # forward 21d simple returns (target at t)
        fwd21 = prices.pct_change(periods=-21)

        # features stacked
        X_long = pd.concat([
            _stack_feature(mu,  'mu'),
            _stack_feature(vol, 'vol'),
            _stack_feature(m6,  'm6'),
            _stack_feature(m12, 'm12'),
        ], axis=1).dropna()

        y_long = fwd21.stack().to_frame('y')
        XY = X_long.join(y_long, how='inner').dropna()

        if XY.empty:
            raise RuntimeError("ML training set is empty after join. Check feature/price alignment.")

        last_date = XY.index.get_level_values(0).max()
        mask_train = XY.index.get_level_values(0) < last_date
        X_train = XY.loc[mask_train, ['mu','vol','m6','m12']].values
        y_train = XY.loc[mask_train, 'y'].values

        # Prediction set for last date (all assets that day)
        X_pred_df = X_long.loc[last_date]
        X_pred = X_pred_df[['mu','vol','m6','m12']].values
        pred_assets = X_pred_df.index.tolist()

        ml_cfg = cfg.get('ml', {})
        models_cfg = ml_cfg.get("models") or [{"name": ml_cfg.get("model", "xgb"), "weight": 1.0}]

        # Train each learner
        trained = []
        total_w = 0.0
        for m in models_cfg:
            name = (m.get("name") or "xgb")
            weight = float(m.get("weight", 1.0))
            model = train_single_model(name, X_train, y_train, ml_cfg)
            trained.append((model, weight))
            total_w += weight

        # Weighted prediction
        preds = np.zeros(X_pred.shape[0])
        total_w = total_w or 1.0
        for model, w in trained:
            preds += w * model.predict(X_pred)
        preds /= total_w

        # Map to full symbol list
        preds_series = pd.Series(preds, index=pred_assets).reindex(symbols).fillna(0.0)

        ml_weights = preds_to_weights(
            preds_series.values,
            cap=cfg['constraints']['max_weight'],
            cash=cfg['constraints']['cash_buffer']
        )

    except Exception as e:
        # Fail-soft: write a breadcrumb, but still produce a file so UI continues
        (PROC / "ml_error.txt").write_text(f"[optimize] ML failed: {e}\n", encoding="utf-8")
        # fallback = small momentum tilt (rank > median)
        r = ranks.values.clip(min=0)
        ml_weights = preds_to_weights(
            r,
            cap=cfg['constraints']['max_weight'],
            cash=cfg['constraints']['cash_buffer']
        )

    # ---------- Collect & save ----------
    weights = pd.DataFrame(
        {
            'MPT': w_mpt,
            'RiskParity': w_rp,
            'Factor': w_factor,
            'ML_XGB': ml_weights,   # keep name for UI compatibility
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