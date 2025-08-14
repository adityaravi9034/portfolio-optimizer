import argparse, yaml
from pathlib import Path
import numpy as np
import pandas as pd
from utils.io import RAW, PROC, load_df, save_df
from utils.metrics import sharpe, sortino, max_drawdown, apply_costs

TRADING_DAYS = 252

def backtest(prices: pd.DataFrame, weights_hist: pd.DataFrame, costs):
    # Simple proportional rebalancing at weight dates
    returns = np.log(prices).diff().fillna(0)
    dates = weights_hist.index
    port_val = []
    w_prev = np.zeros(prices.shape[1])
    cum = 1.0
    turnover_hist = []

    for i in range(1, len(dates)):
        start, end = dates[i-1], dates[i]
        w = weights_hist.loc[start].values
        # Apply costs when we change weights
        factor, turnover = apply_costs(w_prev, w, costs['commission_bps'], costs['slippage_bps'])
        w_prev = w
        period_ret = (returns.loc[start:end] @ w).sum()
        cum *= np.exp(period_ret) * factor
        port_val.append((end, cum))
        turnover_hist.append((end, turnover))

    pv = pd.Series({d:v for d,v in port_val}).sort_index()
    to = pd.Series({d:t for d,t in turnover_hist}).sort_index()
    rets = np.log(pv).diff().dropna()
    metrics = {
        'CAGR': (pv.iloc[-1])**(TRADING_DAYS/len(pv)) - 1,
        'Vol': np.sqrt(TRADING_DAYS) * rets.std(ddof=0),
        'Sharpe': sharpe(rets),
        'Sortino': sortino(rets),
        'MaxDD': max_drawdown(pv)
    }
    return pv, to, metrics

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    prices = load_df(RAW/'prices.csv')
    # For demo, reuse latest weights at monthly interval
    w = pd.read_csv(PROC/'latest_weights.csv', index_col=0)
    # fabricate a monthly schedule using price index
    monthly = prices.resample('M').last().index
    weights_hist = pd.DataFrame([w['MPT'].values]*len(monthly), index=monthly, columns=w.index)

    pv, to, metrics = backtest(prices, weights_hist, cfg['costs'])
    save_df(pv.to_frame('equity'), PROC/'backtest_equity_curve.csv')
    pd.Series(metrics).to_csv(PROC/'latest_metrics.csv')
    print('Saved backtest metrics:', metrics)