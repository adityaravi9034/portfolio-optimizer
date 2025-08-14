import argparse, os, sys, time, yaml
from pathlib import Path
import pandas as pd
import yfinance as yf
import requests
from fredapi import Fred
from utils.io import RAW, save_df

def _drop_tz(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    try:
        return idx.tz_localize(None)
    except Exception:
        return pd.DatetimeIndex(idx)

def fetch_yahoo(symbol: str, period="10y") -> pd.Series:
    for k in range(3):
        try:
            h = yf.Ticker(symbol).history(period=period, auto_adjust=True)
            if "Close" in h and not h["Close"].empty:
                s = h["Close"].rename(symbol)
                s.index = _drop_tz(pd.DatetimeIndex(s.index))
                return s
        except Exception as e:
            if k == 2:
                print(f"[WARN] yfinance {symbol} failed: {e}", file=sys.stderr)
        time.sleep(0.4)
    return pd.Series(name=number, dtype=float)

def fetch_alpha_vantage(symbol: str, fn="TIME_SERIES_DAILY_ADJUSTED") -> pd.Series:
    key = os.getenv("ALPHA_VANTAGE_KEY")
    if not key:
        return pd.Series(name=symbol, dtype=float)
    url = "https://www.alphavantage.co/query"
    params = {"function": fn, "symbol": symbol, "apikey": key, "outputsize": "full"}
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        js = r.json()
        ts = js.get("Time Series (Daily)") or js.get("Time Series FX (Daily)") or {}
        if not ts:
            return pd.Series(name=symbol, dtype=float)
        df = (pd.DataFrame(ts).T
              .rename(columns=lambda c: c.split(". ")[-1].lower())
              .sort_index())
        s = pd.to_numeric(df["adjusted close"] if "adjusted close" in df else df["close"], errors="coerce")
        s.index = pd.to_datetime(s.index)
        s.index = _drop_tz(pd.DatetimeIndex(s.index))
        s.name = symbol
        return s
    except Exception as e:
        print(f"[WARN] alpha_vantage {symbol} failed: {e}", file=sys.stderr)
        return pd.Series(name=symbol, dtype=float)

def fetch_prices(symbols, use_alpha_vantage: bool, av_fn: str):
    series = []
    for s in symbols:
        if use_alpha_vantage and os.getenv("ALPHA_VANTAGE_KEY") and not s.endswith("-USD"):
            ser = fetch_alpha_vantage(s, av_fn)
            if ser.empty:
                ser = fetch_yahoo(s)
        else:
            ser = fetch_yahoo(s)
        series.append(ser)
        time.sleep(0.2)
    df = pd.concat(series, axis=1)
    if not df.empty:
        df = df.sort_index()
        b = pd.bdate_range(df.index.min(), df.index.max())
        df = df.reindex(b).ffill()
    return df

def fetch_fred(fred_map: dict):
    key = os.getenv("FRED_API_KEY")
    if not key:
        print("[ERROR] FRED_API_KEY not set.", file=sys.stderr); sys.exit(3)
    fred = Fred(api_key=key)
    data = {}
    for name, code in fred_map.items():
        try:
            ser = fred.get_series(code)
            s = pd.Series(ser, name=name)
            data[name] = s
        except Exception as e:
            print(f"[WARN] FRED {name}/{code} failed: {e}", file=sys.stderr)
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    df.index = _drop_tz(pd.DatetimeIndex(df.index))
    df = df.sort_index().ffill()
    return df

def main(cfg):
    uni = []
    for k in ("equities","bonds","commodities","crypto"):
        uni.extend(cfg["universe"].get(k, []))
    if not uni:
        print("[ERROR] Empty universe", file=sys.stderr); sys.exit(1)

    use_av = bool(cfg.get("data", {}).get("use_alpha_vantage", False))
    av_fn  = cfg.get("data", {}).get("alpha_vantage_function", "TIME_SERIES_DAILY_ADJUSTED")

    prices = fetch_prices(uni, use_av, av_fn)
    if prices.empty:
        print("[ERROR] No price data", file=sys.stderr); sys.exit(2)
    save_df(prices, RAW / "prices.csv")
    print("Saved prices:", prices.shape, "->", RAW / "prices.csv")

    fred_map = cfg.get("data", {}).get("fred_series", {})
    fred = fetch_fred(fred_map)
    fred = fred.reindex(prices.index, method="ffill")
    save_df(fred, RAW / "macro_fred.csv")
    print("Saved FRED macro:", fred.shape, "->", RAW / "macro_fred.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    main(cfg)
