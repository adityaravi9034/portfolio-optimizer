# pipeline/fetch_market.py
import argparse
import sys
import time
from pathlib import Path
import os
import yaml
import pandas as pd
import numpy as np
import requests
import yfinance as yf

# ======================================================================
# ENVIRONMENT SETUP
# ======================================================================
ROOT = Path(__file__).resolve().parents[1]

def load_environment():
    """Load environment variables with robust fallback"""
    env_path = ROOT / '.env'
    if env_path.exists():
        print(f"[ENV] Loading .env from {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip().strip('"\'')
                    except ValueError:
                        continue

    FRED_API_KEY = os.getenv("FRED_API_KEY")
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
    
    print("\n[ENV] API Key Status:")
    print(f"FRED_API_KEY: {'Set' if FRED_API_KEY else 'Not set'}")
    print(f"ALPHA_VANTAGE_API_KEY: {'Set' if ALPHA_VANTAGE_API_KEY else 'Not set'}")
    
    return FRED_API_KEY, ALPHA_VANTAGE_API_KEY

FRED_API_KEY, _ = load_environment()  # We ignore Alpha Vantage key since we're not using it

# ======================================================================
# DATA FETCHING FUNCTIONS
# ======================================================================
def _drop_tz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Remove timezone from datetime index."""
    return idx.tz_localize(None) if idx.tz else idx

def fetch_yahoo(symbol: str, period: str = "10y") -> pd.Series:
    """Fetch data from Yahoo Finance with retries."""
    for attempt in range(3):
        try:
            print(f"[YAHOO] Fetching {symbol} (attempt {attempt + 1})")
            data = yf.Ticker(symbol).history(period=period, auto_adjust=True)
            if not data.empty and 'Close' in data:
                s = data['Close'].rename(symbol)
                s.index = _drop_tz(s.index)
                return s
        except Exception as e:
            if attempt == 2:
                print(f"[YAHOO ERROR] Failed to fetch {symbol}: {e}")
            time.sleep(1 * (attempt + 1))
    return pd.Series(name=symbol, dtype=float)

def fetch_prices(symbols: list[str]) -> pd.DataFrame:
    """Fetch all prices using Yahoo Finance."""
    series = []
    for symbol in symbols:
        series.append(fetch_yahoo(symbol))
        time.sleep(0.2)  # Be polite to Yahoo's servers
    df = pd.concat(series, axis=1)
    if not df.empty:
        df = df.sort_index()
        bdays = pd.bdate_range(df.index.min(), df.index.max())
        df = df.reindex(bdays).ffill()
    return df

def fetch_fred(series_map: dict[str, str], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Fetch FRED economic data."""
    if not FRED_API_KEY or not series_map:
        return pd.DataFrame(index=index)
    
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        data = {}
        for name, code in series_map.items():
            try:
                print(f"[FRED] Fetching {name} ({code})")
                data[name] = fred.get_series(code)
            except Exception as e:
                print(f"[FRED ERROR] Failed to fetch {name}: {e}")
        
        if data:
            df = pd.DataFrame(data)
            df.index = _drop_tz(pd.to_datetime(df.index))
            return df.reindex(index, method='ffill')
    except Exception as e:
        print(f"[FRED ERROR] API failure: {e}")
    
    return pd.DataFrame(index=index)

# ======================================================================
# MAIN FUNCTION
# ======================================================================
def main(cfg: dict):
    # Load tickers from config
    tickers = []
    for asset_class in ["equities", "bonds", "commodities", "crypto"]:
        tickers.extend(cfg.get("universe", {}).get(asset_class, []))
    
    if not tickers:
        from utils.ticks import full_universe
        tickers = full_universe()
    
    print(f"\n[MAIN] Fetching {len(tickers)} assets")
    
    # Fetch prices
    prices = fetch_prices(tickers)
    if prices.empty:
        print("[ERROR] No price data retrieved")
        sys.exit(1)
    
    # Save prices
    from utils.io import RAW, save_df
    save_df(prices, RAW / "prices.csv")
    print(f"[SAVED] Prices: {prices.shape}")
    
    # Fetch FRED data
    fred_data = fetch_fred(
        cfg.get("data", {}).get("fred_series", {}),
        prices.index
    )
    save_df(fred_data, RAW / "macro_fred.csv")
    print(f"[SAVED] FRED Data: {fred_data.shape if not fred_data.empty else 'Empty'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    try:
        cfg = yaml.safe_load(Path(args.config).read_text())
        main(cfg)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)