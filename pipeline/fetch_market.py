import argparse, yaml
from pathlib import Path
import pandas as pd
import yfinance as yf
from fredapi import Fred
from utils.io import RAW, save_df

FRED_SERIES = {
    'CPI': 'CPIAUCSL',
    'GDP': 'GDPC1',
    'UNRATE': 'UNRATE',
    'DGS3MO': 'DGS3MO',
    'DGS2': 'DGS2',
    'DGS10': 'DGS10'
}

def fetch_prices(symbols):
    data = yf.download(symbols, auto_adjust=True, progress=False)['Close']
    return data.dropna(how='all')

def fetch_fred():
    fred = Fred()  # no API key needed for many series; optional env var FRED_API_KEY
    df = pd.DataFrame({k: fred.get_series(v) for k, v in FRED_SERIES.items()})
    df.index = pd.to_datetime(df.index)
    return df

def main(cfg):
    uni = []
    for k in ('equities','bonds','commodities','crypto'):
        uni.extend(cfg['universe'].get(k, []))
    prices = fetch_prices(uni)
    save_df(prices, RAW/ 'prices.csv')

    fred = fetch_fred()
    save_df(fred, RAW/ 'macro_fred.csv')
    print('Fetched:', prices.shape, fred.shape)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    main(cfg)