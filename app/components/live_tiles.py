# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
# app/components/live_tiles.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

def _load_prices():
    p = RAW / "prices.csv"
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        return df
    except Exception:
        return None

def _load_equity():
    p = PROC / "equity_curves.csv"
    if not p.exists() or p.stat().st_size == 0:
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, errors="coerce")
    return df[~df.index.isna()]

def render_live_kpis(top_n: int = 6):
    st.caption("Live tiles refresh with the page (see Sidebar ▸ Auto-refresh)")
    prices = _load_prices()
    eq = _load_equity()

    colz = st.columns(3)
    # Portfolio snapshot
    with colz[0]:
        st.markdown("**📈 Portfolio snapshot**")
        if eq is not None and not eq.empty:
            latest = eq.iloc[-1]
            best = latest.idxmax()
            st.metric("Best model (equity x)", f"{latest[best]:.2f}x", help=f"Top: {best}")
        else:
            st.info("Run walkforward/backtest for equity curves.")

    # Last updated
    with colz[1]:
        st.markdown("**🗓️ Data timestamp**")
        if prices is not None and not prices.empty:
            st.metric("Last bar", prices.index[-1].strftime("%Y-%m-%d"))
        else:
            st.info("No prices yet.")

    # Universe breadth
    with colz[2]:
        st.markdown("**🧺 Universe**")
        if prices is not None:
            st.metric("Assets", f"{prices.shape[1]}")
        else:
            st.metric("Assets", "—")

    st.markdown("---")
    st.markdown("**Top movers (1D)**")
    if prices is None or prices.shape[0] < 2:
        st.info("Need at least two rows of prices.")
        return

    last = prices.iloc[-1]
    prev = prices.iloc[-2]
    chg = (last / prev - 1.0).sort_values(ascending=False)

    up = chg.head(top_n)
    dn = chg.tail(top_n).sort_values()

    cu, cd = st.columns(2)
    with cu:
        st.write("▲ Gainers")
        st.dataframe(up.to_frame("1D %").style.format("{:+.2%}"), use_container_width=True)
    with cd:
        st.write("▼ Losers")
        st.dataframe(dn.to_frame("1D %").style.format("{:+.2%}"), use_container_width=True)