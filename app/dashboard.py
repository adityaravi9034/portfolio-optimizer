import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
PROC = Path(__file__).resolve().parents[1] / 'data' / 'processed'

st.title("📈 Multi-Asset Portfolio Optimizer (Free Tier)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Models", "Backtest", "Stress", "Signals"])

# ---------- Overview ----------
with tab1:
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("Latest Portfolio Weights")
        wp = PROC / "latest_weights.csv"
        if wp.exists():
            w = pd.read_csv(wp, index_col=0)
            st.dataframe(w.style.format("{:.2%}"))
            st.download_button("Download Weights CSV", data=w.to_csv().encode(), file_name="latest_weights.csv")
        else:
            st.info("No weights yet. Run the daily pipeline.")
    with col2:
        st.subheader("Latest Metrics")
        mp = PROC / "latest_metrics.csv"
        if mp.exists():
            m = pd.read_csv(mp, index_col=0, header=None).squeeze()
            st.dataframe(m.to_frame("value"))
        else:
            st.info("No metrics yet. Run backtest.")

# ---------- Models ----------
with tab2:
    st.subheader("Per-Model KPIs")
    mp = PROC / "metrics_by_model.csv"
    if mp.exists():
        k = pd.read_csv(mp, index_col=0)
        st.dataframe(k.style.format("{:.2%}", subset=["CAGR","Vol","MaxDD"]).format("{:.2f}", subset=["Sharpe","Sortino"]))
    else:
        st.info("Run walk-forward to compute per-model KPIs.")

    st.subheader("Equity Curves")
    ep = PROC / "equity_curves.csv"
    if ep.exists():
        eq = pd.read_csv(ep, index_col=0, parse_dates=True)
        st.line_chart(eq)
    else:
        st.info("Run walk-forward to build equity curves.")

# ---------- Backtest ----------
with tab3:
    st.subheader("Weights Over Time (Latest Walk-Forward)")
    wt = PROC / "weights_timeseries.csv"
    if wt.exists():
        wt_df = pd.read_csv(wt, index_col=0, parse_dates=True)
        model = st.selectbox("Model", ["MPT","RiskParity","Factor","ML_XGB"], index=0)
        # columns are assets; same for each model panel
        st.area_chart(wt_df)  # simple view: overall stacked weights
        st.caption("Tip: filter columns in the table below to inspect per-asset weights.")
        st.dataframe(wt_df.tail(24).style.format("{:.2%}"))
    else:
        st.info("Run walk-forward to generate weights_timeseries.csv.")

# ---------- Stress ----------
with tab4:
    st.subheader("Historical Stress Periods")
    sp = PROC / "stress_summary.csv"
    if sp.exists():
        s = pd.read_csv(sp)
        st.dataframe(s)
    else:
        st.info("Run `python -m pipeline.stress --config configs/default.yaml` to generate stress summary.")

# ---------- Signals ----------
with tab5:
    st.subheader("Rebalance Recommendations (Diff vs Previous Weights)")
    wp = PROC / "latest_weights.csv"
    if wp.exists():
        lw = pd.read_csv(wp, index_col=0)
        # synthesize "previous weights" as zeros for first run; or cache last run to data/processed/prev_weights.csv
        prevp = PROC / "prev_weights.csv"
        if prevp.exists():
            prev = pd.read_csv(prevp, index_col=0)
        else:
            prev = lw.copy()*0.0
        model = st.selectbox("Model", lw.columns.to_list(), index=0)
        delta = (lw[model] - prev[model]).to_frame("Δweight").sort_values("Δweight")
        st.dataframe(delta.style.format("{:+.2%}"))
        st.caption("Positive = buy, negative = sell. Apply your transaction cost model at execution.")
        # update prev for next run (local session)
        prev.to_csv(PROC / "prev_weights.csv")
    else:
        st.info("No latest weights yet.")

st.markdown("---")
st.caption("Run: `python -m pipeline.fetch_market && python -m pipeline.build_features && python -m pipeline.optimize && python -m pipeline.walkforward && python -m pipeline.backtest`")
