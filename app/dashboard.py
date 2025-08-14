import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

st.title("📈 Multi-Asset Portfolio Optimizer (Free Tier)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview", "Models", "Backtest", "Stress", "Signals"])

# ---------- Helpers ----------
def load_csv_indexed(path: Path):
    """Read a CSV with first column as datetime index (if any); fall back gracefully."""
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    # try to parse index as dates; ignore if it fails
    try:
        df.index = pd.to_datetime(df.index, utc=False)
    except Exception:
        pass
    return df

# ---------- Overview ----------
with tab1:
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("Latest Portfolio Weights")
        wp = PROC / "latest_weights.csv"
        w = load_csv_indexed(wp)
        if w is not None:
            st.dataframe(w.style.format("{:.2%}"))
            st.download_button(
                "Download Weights CSV",
                data=w.to_csv().encode(),
                file_name="latest_weights.csv",
            )
        else:
            st.info("No weights yet. Run the daily pipeline.")

    with col2:
        st.subheader("Latest Metrics")
        mp = PROC / "latest_metrics.csv"
        if mp.exists():
            # latest_metrics.csv is a two-column format from pipeline
            m = pd.read_csv(mp, index_col=0, header=None).squeeze()
            st.dataframe(m.to_frame("value"))
            st.download_button(
                "Download Metrics CSV",
                data=pd.DataFrame(m).to_csv().encode(),
                file_name="latest_metrics.csv",
            )
        else:
            st.info("No metrics yet. Run backtest.")

# ---------- Models ----------
with tab2:
    st.subheader("Per-Model KPIs")
    kpi_path = PROC / "metrics_by_model.csv"
    k = load_csv_indexed(kpi_path)
    if k is not None:
        st.dataframe(
            k.style.format("{:.2%}", subset=["CAGR", "Vol", "MaxDD"]).format(
                "{:.2f}", subset=["Sharpe", "Sortino"]
            )
        )
        st.download_button(
            "Download Per-Model KPIs",
            data=k.to_csv().encode(),
            file_name="metrics_by_model.csv",
        )
    else:
        st.info("Run walk-forward to compute per-model KPIs (see README).")

    st.subheader("Equity Curves")
    ep = PROC / "equity_curves.csv"
    eq = load_csv_indexed(ep)
    if eq is not None:
        st.line_chart(eq)

        # Drawdowns
        st.subheader("Drawdowns")
        dd = eq / eq.cummax() - 1.0
        st.line_chart(dd)
        st.caption("Drawdown = Equity / Running Max − 1")
    else:
        st.info("Run walk-forward to build equity curves.")

# ---------- Backtest ----------
with tab3:
    st.subheader("Weights Over Time (Latest Walk-Forward)")
    wt = PROC / "weights_timeseries.csv"
    wt_df = load_csv_indexed(wt)
    if wt_df is not None:
        model = st.selectbox(
            "Model",
            ["MPT", "RiskParity", "Factor", "ML_XGB"],
            index=0,
            key="model_backtest",
        )
        # Simple overview (stacked) — shows total allocation across assets
        st.area_chart(wt_df)
        st.caption("Tip: Use the table to inspect last 24 months of per-asset weights.")
        st.dataframe(wt_df.tail(24).style.format("{:.2%}"))
        st.download_button(
            "Download Weights Timeseries",
            data=wt_df.to_csv().encode(),
            file_name="weights_timeseries.csv",
        )
    else:
        st.info("Run walk-forward to generate weights_timeseries.csv.")

# ---------- Stress ----------
with tab4:
    st.subheader("Historical Stress Periods")
    sp = PROC / "stress_summary.csv"
    s = load_csv_indexed(sp) if sp.exists() else None
    if s is not None:
        st.dataframe(s)
        st.download_button(
            "Download Stress Summary",
            data=s.to_csv().encode(),
            file_name="stress_summary.csv",
        )
    else:
        st.info("Run `python -m pipeline.stress --config configs/default.yaml` to generate stress summary.")

# ---------- Signals ----------
with tab5:
    st.subheader("Rebalance Recommendations")
    wp = PROC / "latest_weights.csv"
    lw = load_csv_indexed(wp)
    trades_dir = PROC / "trades"

    if lw is not None:
        model = st.selectbox("Model", lw.columns.to_list(), index=0, key="model_signals")

        # Previous weights (persisted by trades pipeline); zero if first run
        prevp = PROC / "prev_weights.csv"
        prev = load_csv_indexed(prevp)
        if prev is None:
            prev = lw.copy() * 0.0

        # Delta weights
        delta = (lw[model] - prev.get(model, 0.0)).to_frame("Δweight").sort_values("Δweight")
        st.dataframe(delta.style.format("{:+.2%}"))
        st.caption("Positive = buy, negative = sell. Apply your transaction cost model at execution.")

        st.markdown("### Trade Blotter")
        trades_file = trades_dir / f"trades_{model}.csv"
        if trades_file.exists():
            tb = pd.read_csv(trades_file)
            st.dataframe(
                tb.style.format(
                    {
                        "delta_weight": "{:+.2%}",
                        "dollars": "{:+,.0f}",
                        "last_price": "{:,.2f}",
                        "approx_shares": "{:+.3f}",
                    }
                )
            )
            st.download_button(
                label=f"Download {model} trades CSV",
                data=tb.to_csv(index=False).encode(),
                file_name=f"trades_{model}.csv",
            )
        else:
            st.info("No trade blotter yet. Run: `python -m pipeline.trades --config configs/default.yaml`")

        # (Optional) store prev for the next run if not present
        if not prevp.exists():
            lw.to_csv(prevp)
    else:
        st.info("No latest weights yet. Generate them via the optimize step.")

st.markdown("---")
st.caption(
    "Run locally: "
    "`python -m pipeline.fetch_market && python -m pipeline.build_features && "
    "python -m pipeline.optimize && python -m pipeline.walkforward && "
    "python -m pipeline.stress && python -m pipeline.backtest && "
    "python -m pipeline.trades -- --notional 100000`"
)
