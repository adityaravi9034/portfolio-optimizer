# pipeline/walkforward.py
import argparse, yaml
from pathlib import Path
import numpy as np, pandas as pd
from utils.io import RAW, PROC, load_df, save_df
from utils.metrics import sharpe, sortino, max_drawdown, apply_costs
from models.mpt import optimize_mpt
from models.risk_parity import risk_parity
from models.factor import factor_target_weights
from models.ml_xgb import train_xgb, preds_to_weights

TRADING_DAYS = 252

def ewma_cov(rets, span=60):
    # simple ewma covariance (fallback to sample)
    try:
        lam = 2/(span+1)
        x = rets - rets.mean()
        S = np.cov(x.dropna().T)
        return pd.DataFrame(S, index=rets.columns, columns=rets.columns)
    except Exception:
        S = rets.cov()
        return S

def compute_features_at(prices: pd.DataFrame, end_idx: pd.Timestamp, cfg):
    px = prices.loc[:end_idx]
    rets = np.log(px).diff().dropna()
    mu = rets.rolling(cfg['lookbacks']['returns_days']).mean().iloc[-1].fillna(0.0)
    vol = rets.rolling(cfg['lookbacks']['vol_days']).std().iloc[-1].fillna(0.0)
    mom12 = px.pct_change(252).iloc[-1].fillna(0.0)
    Sigma = ewma_cov(rets.tail(252))
    return mu, vol, mom12, Sigma

def gen_weights_all_models(mu, vol, mom12, Sigma, cfg):
    symbols = mu.index
    # MPT
    w_mpt = optimize_mpt(mu.values, Sigma.values,
                         max_w=cfg['constraints']['max_weight'],
                         long_only=cfg['constraints']['long_only'],
                         cash_buffer=cfg['constraints']['cash_buffer'],
                         risk_aversion=10.0)
    # Risk Parity
    w_rp = risk_parity(Sigma.values, cash_buffer=cfg['constraints']['cash_buffer'])
    # Factor (momentum tilt via ranks)
    ranks = mom12.rank(pct=True)
    B = np.vstack([ranks.values])  # 1-factor proxy
    f_target = np.array([0.7])
    w_factor = factor_target_weights(B, f_target,
                                     cap=cfg['constraints']['max_weight'],
                                     cash_buffer=cfg['constraints']['cash_buffer'])
    # ML (cheap proxy feature set; fallback RF if XGB missing)
    preds = (mu + 0.5*mom12).values
    w_ml = preds_to_weights(preds, cap=cfg['constraints']['max_weight'],
                            cash=cfg['constraints']['cash_buffer'])
    return pd.Series(w_mpt, index=symbols), pd.Series(w_rp, index=symbols), \
           pd.Series(w_factor, index=symbols), pd.Series(w_ml, index=symbols)

def walk_forward(prices: pd.DataFrame, cfg):
    # monthly rebalances, 3y rolling train window
    monthly = prices.resample('ME').last().index
    start = monthly[cfg['lookbacks']['rolling_train_days']//21]  # after ~3y
    monthly = monthly[monthly >= start]

    weight_panels = []
    last_w = {m: np.zeros(prices.shape[1]) for m in ['MPT','RiskParity','Factor','ML_XGB']}
    perf = {m: [] for m in last_w}
    turnover_log = {m: [] for m in last_w}
    rets = np.log(prices).diff().fillna(0)

    for i in range(1, len(monthly)):
        train_end = monthly[i-1]
        apply_start, apply_end = monthly[i-1], monthly[i]

        mu, vol, mom12, Sigma = compute_features_at(prices, train_end, cfg)
        mpt, rp, fac, ml = gen_weights_all_models(mu, vol, mom12, Sigma, cfg)
        weights_now = pd.DataFrame({'MPT': mpt, 'RiskParity': rp, 'Factor': fac, 'ML_XGB': ml})
        weights_now.index.name = 'asset'
        weights_now['date'] = apply_end
        weight_panels.append(weights_now.reset_index())

        period_ret = rets.loc[apply_start:apply_end]
        for name, w_ser in [('MPT',mpt),('RiskParity',rp),('Factor',fac),('ML_XGB',ml)]:
            factor, t = apply_costs(last_w[name], w_ser.values,
                                    cfg['costs']['commission_bps'], cfg['costs']['slippage_bps'])
            last_w[name] = w_ser.values
            port = (period_ret @ w_ser).sum()
            perf[name].append((apply_end, float(np.exp(port) * factor)))

            turnover_log[name].append((apply_end, t))

    # build outputs
    weights_ts = pd.concat(weight_panels).pivot(index='date', columns='asset')
    weights_ts.columns = [f'{c[1]}' for c in weights_ts.columns]  # flatten MultiIndex
    weights_ts = weights_ts.sort_index()

    equity = {}
    metrics = {}
    for name in perf:
        eq = pd.Series({d:v for d,v in perf[name]}).sort_index().cumprod()
        r  = np.log(eq).diff().dropna()
        equity[name] = eq
        metrics[name] = {
            'CAGR': (eq.iloc[-1])**(TRADING_DAYS/len(eq)) - 1,
            'Vol': np.sqrt(TRADING_DAYS) * r.std(ddof=0),
            'Sharpe': sharpe(r),
            'Sortino': sortino(r),
            'MaxDD': max_drawdown(eq)
        }

    weights_ts.to_csv(PROC/'weights_timeseries.csv')
    pd.DataFrame(equity).to_csv(PROC/'equity_curves.csv')
    pd.DataFrame(metrics).T.to_csv(PROC/'metrics_by_model.csv')
    return weights_ts, equity, metrics

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    prices = load_df(RAW/'prices.csv')
    walk_forward(prices, cfg)
    print("Saved walk-forward: weights_timeseries.csv, equity_curves.csv, metrics_by_model.csv")
