# app/components/reports.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import requests
import streamlit as st


def render_reports_panel(api_base: str, use_api: bool, root: Path, key_prefix: str | None = None):
    # ---- ensure unique keys if caller didn't pass a prefix ----
    if key_prefix is None:
        seq_key = "_reports_seq"
        seq = st.session_state.get(seq_key, 0)
        st.session_state[seq_key] = seq + 1
        key_prefix = f"reports_{seq}"

    st.markdown("### 📄 Reports")

    # --- Generate report button ---
    gen_key = f"{key_prefix}_generate"
    if st.button("Generate daily report", type="primary", use_container_width=True, key=gen_key):
        if not use_api:
            st.warning("API mode is off. Set OPTIM_API_URL and restart.")
        else:
            try:
                with st.spinner("Generating…"):
                    r = requests.post(f"{api_base}/report/daily", timeout=30)
                    r.raise_for_status()
                    out = r.json().get("path")
                    st.success(f"Report created: {out}")
            except Exception as e:
                st.error(f"Report failed: {e}")

    st.caption("Latest files in data/reports/")

    # --- List recent reports ---
    reports_dir = root / "data" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(reports_dir.glob("report_*.html"), reverse=True)[:20]

    if not files:
        st.info("No reports yet. Generate one above.")
    else:
        for i, f in enumerate(files):
            ts = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            col1, col2, col3 = st.columns([4, 2, 2])
            with col1:
                st.write(f"**{f.name}**  \n_{ts}_")
            with col2:
                # link avoids making more Streamlit buttons
                st.markdown(f"[Open in browser](file://{f.resolve()})")
            with col3:
                st.download_button(
                    "Download",
                    data=f.read_bytes(),
                    file_name=f.name,
                    mime="text/html",
                    key=f"{key_prefix}_dl_{i}",
                    use_container_width=True,
                )

    # --- Test email button (INSIDE the function) ---
    test_key = f"{key_prefix}_send_test"
    if st.button("Send Test Email", key=test_key):
        if not use_api:
            st.warning("API mode is off. Set OPTIM_API_URL and restart.")
        else:
            try:
                resp = requests.post(f"{api_base}/email/test", timeout=15)
                if resp.ok and (resp.json().get("ok") is True):
                    st.success("✅ Test email sent! Check your inbox.")
                else:
                    st.error(f"❌ Failed: {resp.text}")
            except Exception as e:
                st.error(f"Error: {e}")