# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
from __future__ import annotations
import streamlit as st, yaml
from pathlib import Path
import pandas as pd
from io import StringIO

def render_universe_editor(cfg_path: Path):
    st.markdown("#### Upload tickers (CSV, TXT, or paste)")
    upload = st.file_uploader("File with one ticker per line or column", type=["csv","txt"])
    pasted = st.text_area("…or paste tickers (comma/space/newline separated)")

    tickers: list[str] = []

    if upload is not None:
        data = upload.read().decode("utf-8", errors="ignore")
        try:
            df = pd.read_csv(StringIO(data))
            # take first column
            tickers = [str(t).strip() for t in df.iloc[:,0].dropna().astype(str)]
        except Exception:
            # fallback: split lines
            tickers = [t.strip() for t in data.splitlines() if t.strip()]
    elif pasted.strip():
        raw = pasted.replace(",", " ").split()
        tickers = [t.strip() for t in raw if t.strip()]

    if tickers:
        st.write(f"Detected **{len(tickers)}** tickers")
        st.dataframe(pd.DataFrame({"Ticker": tickers}), use_container_width=True, height=240)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Apply to config", type="primary", disabled=not tickers, key="univ_apply"):
            cfg = yaml.safe_load(cfg_path.read_text()) or {}
            cfg.setdefault("universe", {})["equities"] = tickers
            cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
            st.success("Universe updated in YAML")
            st.session_state["universe_last"] = tickers

    with col2:
        if st.button("Run Fetch + Features", disabled=not tickers, key="univ_run"):
            st.session_state["trigger_run_fetch_features"] = True