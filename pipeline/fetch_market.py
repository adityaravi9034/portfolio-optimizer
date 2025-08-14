import argparse, os, sys, time, yaml
from pathlib import Path
import pandas as pd
import yfinance as yf
from fredapi import Fred
from utils.io import RAW, save_df

FRED_SERIES = {
    "CPI": "CPIAUCSL",
    "GDP": "GDPC1",
    "UNRATE": "UNRATE",
    "DGS3MO": "DGS3MO",
    "DGS2": "DGS2",
    "DGS10": "DGS10",
}

def _drop_tz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # Make DateTimeIndex tz-naive regardless of source
    try:
        return idx.tz_localize(None)
    except (TypeError, AttributeError, ValueError):
        # already naive or not tz-aware
        return pd.DatetimeIndex(idx)

def fetch_one(sym: str, period="10y"):
    for k in range(3):
        try:
            h = yf.Ticker(sym).history(period=period, auto_adjust=True)
            if "Close" in h and not h["Close"].empty:
                s = h["Close"].rename(sym)
                # yfinance returns tz-aware (UTC); drop tz to align with FRED
                s.index = _drop_tz(pd.DatetimeIndex(s.index))
                return s
        except Exception as e:
            if k == 2:
                print(f"[WARN] {sym} failed: {e}", file=sys.stderr)
        time.sleep(0.4)
    return pd.Series(name=sym, dtype=float)

def fetch_prices(symbols):
    series = [fetch_one(s) for s in symbols]
    df = pd.concat(series, axis=1)
    if not df.empty:
        df = df.sort_index()
        b = pd.bdate_range(df.index.min(), df.index.max())
        df = df.reindex(b).ffill()
    return df

def fetch_fred():
    key = os.getenv("FRED_API_KEY")
    if not key:
        print("[ERROR] FRED_API_KEY not set in environment.", file=sys.stderr)
        sys.exit(3)
    fred = Fred(api_key=key)
    data = {}
    for name, code in FRED_SERIES.items():
        ser = fred.get_series(code)
        s = pd.Series(ser, name=name)
        data[name] = s
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    df.index = _drop_tz(pd.DatetimeIndex(df.index))
    df = df.sort_index().ffill()
    return df

def main(cfg):
    # --- PRICES ---
    uni = []
    for k in ("equities","bonds","commodities","crypto"):
        uni.extend(cfg["universe"].get(k, []))
    if not uni:
        print("[ERROR] Empty universe in config.", file=sys.stderr); sys.exit(1)

    prices = fetch_prices(uni)
    if prices.empty:
        print("[ERROR] yfinance returned no data.", file=sys.stderr); sys.exit(2)
    save_df(prices, RAW / "prices.csv")
    print("Saved prices:", prices.shape, "->", RAW / "prices.csv")

    # --- FRED ---
    fred = fetch_fred()
    # normalize indexes (defensive)
    prices.index = _drop_tz(pd.DatetimeIndex(prices.index))
    fred.index   = _drop_tz(pd.DatetimeIndex(fred.index))
    fred_b = fred.reindex(prices.index, method="ffill")
    save_df(fred_b, RAW / "macro_fred.csv")
    print("Saved FRED macro:", fred_b.shape, "->", RAW / "macro_fred.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    main(cfg)
