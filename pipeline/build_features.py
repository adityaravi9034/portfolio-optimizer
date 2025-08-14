import argparse, yaml
from pathlib import Path
import numpy as np
import pandas as pd
from utils.io import RAW, PROC, load_df, save_df

TRADING_DAYS = 252

def build_asset_features(prices: pd.DataFrame, cfg):
    ret = np.log(prices).diff().dropna()
    # Rolling stats
    mu = ret.rolling(cfg['lookbacks']['returns_days']).mean()
    vol = ret.rolling(cfg['lookbacks']['vol_days']).std() * np.sqrt(TRADING_DAYS)
    mom_6m = prices.pct_change(126)
    mom_12m = prices.pct_change(252)
    feats = {
        'mu': mu,
        'vol': vol,
        'mom6': mom_6m,
        'mom12': mom_12m
    }
    return feats

def build_macro_features(fred: pd.DataFrame):
    df = fred.copy().sort_index().ffill()
    df['CPI_YoY'] = df['CPI'].pct_change(12)
    df['YC_2s10s'] = df['DGS10'] - df['DGS2']
    df['YC_3m10y'] = df['DGS10'] - df['DGS3MO']
    return df

def main(cfg):
    prices = load_df(RAW/'prices.csv')
    fred = load_df(RAW/'macro_fred.csv')

    feats = build_asset_features(prices, cfg)
    macro = build_macro_features(fred)

    # Align macro to prices frequency (forward-fill to business days)
    macro_b = macro.reindex(prices.index, method='ffill')

    # Save
    for k, v in feats.items():
        save_df(v, PROC/f'feat_{k}.csv')
    save_df(macro_b, PROC/'feat_macro.csv')

    print('Features saved:', [f'feat_{k}.csv' for k in feats], 'feat_macro.csv')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    main(cfg)