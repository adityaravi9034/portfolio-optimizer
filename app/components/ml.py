# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
from __future__ import annotations
from pathlib import Path
import pandas as pd
import requests
import streamlit as st

def _load_csv(p: Path) -> pd.DataFrame | None:
    if not p.exists() or p.stat().st_size == 0:
        return None
    return pd.read_csv(p)

def render_ml_panel(api_base: str, use_api: bool, root: Path, key_prefix: str = "ml"):
    st.subheader("ML Predictions (next ~21d)")
    proc = root / "data" / "processed"

    # --- load ---
    df_preds = None
    df_imp   = None
    if use_api and api_base:
        try:
            r = requests.get(f"{api_base.rstrip('/')}/ml/preds", timeout=5)
            if r.ok:
                df_preds = pd.DataFrame(r.json().get("items", []))
            r2 = requests.get(f"{api_base.rstrip('/')}/ml/importance", timeout=5)
            if r2.ok:
                df_imp = pd.DataFrame(r2.json().get("items", []))
        except Exception as e:
            st.warning(f"API load failed: {e}")
    else:
        df_preds = _load_csv(proc / "ml_latest_preds.csv")
        df_imp   = _load_csv(proc / "ml_feature_importance.csv")

    # --- UI ---
    if df_preds is None or df_preds.empty:
        st.info("No predictions yet. Run Features → Optimize (with ML) to generate `ml_latest_preds.csv`.")
        return

    st.dataframe(
        df_preds.style.format({"pred_21d": "{:+.2%}", "rank": "{:.2f}"}),
        use_container_width=True, height=360
    )

    # Top & bottom picks
    n = st.slider("Show top/bottom N", 3, 15, 5, key=f"{key_prefix}_N")
    top = df_preds.nlargest(n, "pred_21d")
    bot = df_preds.nsmallest(n, "pred_21d")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top predicted**")
        st.bar_chart(top.set_index("asset")["pred_21d"])
    with col2:
        st.markdown("**Bottom predicted**")
        st.bar_chart(bot.set_index("asset")["pred_21d"])

    st.download_button(
        "⬇️ Download predictions CSV",
        data=df_preds.to_csv(index=False).encode(),
        file_name="ml_latest_preds.csv",
        key=f"{key_prefix}_dl_preds",
    )

    st.markdown("---")
    st.markdown("**Feature importance**")
    if df_imp is None or df_imp.empty:
        st.caption("No feature importance available for this model.")
    else:
        st.dataframe(df_imp, use_container_width=True, height=240)
        st.bar_chart(df_imp.set_index("feature")["importance"])