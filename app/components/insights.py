# app/components/insights.py
from __future__ import annotations
import os
import requests
import streamlit as st

API = os.getenv("OPTIM_API_URL", "").rstrip("/")

def render_insights(api_base: str, use_api: bool = False, key_prefix: str = ""):
    st.subheader("🔎 AI-ish Insights")
    if not API:
        st.info("Enable API mode (set OPTIM_API_URL) to fetch insights.")
        return
    try:
        r = requests.get(f"{API}/insights", timeout=10)
        r.raise_for_status()
        items = r.json().get("items", [])
        if not items:
            st.info("No insights yet — run the pipeline first.")
            return
        for bullet in items:
            st.markdown(f"- {bullet}")
    except Exception as e:
        st.error(f"Insights failed: {e}")