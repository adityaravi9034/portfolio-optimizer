import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

PROC = Path(__file__).resolve().parents[1] / 'data' / 'processed'

st.title("📈 Multi-Asset Portfolio Optimizer (Free Tier)")

weights_p = PROC/ 'latest_weights.csv'
metrics_p = PROC/ 'latest_metrics.csv'

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Latest Portfolio Weights")
    if weights_p.exists():
        w = pd.read_csv(weights_p, index_col=0)
        st.dataframe(w.style.format("{:.2%}"))
        st.download_button("Download Weights CSV", data=w.to_csv().encode('utf-8'), file_name='latest_weights.csv')
    else:
        st.info("No weights yet. Run the daily pipeline.")

with col2:
    st.subheader("Latest Metrics (Demo)")
    if metrics_p.exists():
        m = pd.read_csv(metrics_p, index_col=0, header=None).squeeze()
        st.write(m)
    else:
        st.info("No metrics yet. Run backtest.")

st.markdown("---")

st.subheader("How to update")
st.code("""
# From repo root:
python scripts/run_daily.py --config configs/default.yaml
streamlit run app/dashboard.py
""", language='bash')