import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# --- Add project root to sys.path so `utils` can be imported ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.io import load_df  # after sys.path update

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

PROC = ROOT / "data" / "processed"

st.title("📈 Multi-Asset Portfolio Optimizer (Free Tier)")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Models", "Backtest", "Stress", "Signals", "Attribution"]
)

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# ... your sys.path / ROOT / PROC setup stays as you already have it ...

@st.cache_data
def cached_load_df(path: str, *, ts: bool = False, multi: bool = False, index_col: int = 0):
    """
    Cached CSV loader.
      - ts=True       → parse index as datetime (robust).
      - multi=True    → try reading columns as a 2-level MultiIndex.
      - index_col     → index column (default 0).
    Returns None if file is missing.
    """
    p = Path(path)
    if not p.exists():
        return None

    if multi:
        # Try MultiIndex columns; fall back to simple
        try:
            df = pd.read_csv(p, header=[0, 1], index_col=index_col)
            if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
                df = pd.read_csv(p, index_col=index_col)
        except Exception:
            df = pd.read_csv(p, index_col=index_col)
    else:
        df = pd.read_csv(p, index_col=index_col)

    if ts:
        # Fast path first; fallback to general parser if needed
        try:
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
            if df.index.isna().any():
                df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]

    return df
# ===== Sidebar controls =====
import streamlit as st
import yaml
from pathlib import Path

from services.orchestrator import (
    run_stage, run_fetch, run_features, run_optimize,
    run_walkforward, run_stress, run_backtest, run_attribution, run_all
)

CFG_PATH = Path("configs/default.yaml")

st.sidebar.markdown("### ⚙️ Pipeline controls")
col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Run All"):
        run_stage(run_all, "Full pipeline")
        st.rerun()
with col_b:
    if st.button("Refresh Data"):
        run_stage(run_fetch, "Fetch market data")
        run_stage(run_features, "Build features")
        st.rerun()

with st.sidebar.expander("Run a specific stage"):
    if st.button("Optimize → Weights"):
        run_stage(run_optimize, "Optimize")
        st.rerun()
    if st.button("Walkforward"):
        run_stage(run_walkforward, "Walkforward")
        st.rerun()
    if st.button("Backtest"):
        run_stage(run_backtest, "Backtest")
        st.rerun()
    if st.button("Stress"):
        run_stage(run_stress, "Stress")
        st.rerun()
    if st.button("Attribution"):
        run_stage(run_attribution, "Attribution")
        st.rerun()

st.sidebar.markdown("### 🧰 Strategy settings")
with st.sidebar.form("strategy_form"):
    # Universe
    tickers_default = (
        "SPY,QQQ,EFA,EEM,IWM,VNQ,XLK,XLF,XLE,XLV,XLB,XLY,XLP,"
        "IEF,TLT,LQD,HYG,SHY,TIP,BNDX,GLD,SLV,DBC,USO,BTC-USD,ETH-USD"
    )
    tickers_str = st.text_area(
        "Universe (comma-separated tickers)",
        value=tickers_default,
        height=100,
    )

    # Constraints
    max_w = st.slider("Max weight per asset", 0.05, 1.0, 0.30, 0.05)
    long_only = st.checkbox("Long only", True)
    cash_buffer = st.slider("Cash buffer", 0.0, 0.20, 0.00, 0.01)

    submitted = st.form_submit_button("Save Settings")
    if submitted:
        cfg = yaml.safe_load(CFG_PATH.read_text())
        # write universe into config
        u = [s.strip() for s in tickers_str.split(",") if s.strip()]
        cfg.setdefault("universe", {})
        # keep one bucket for simplicity
        cfg["universe"]["equities"] = u
        # constraints
        cfg.setdefault("constraints", {})
        cfg["constraints"]["max_weight"] = float(max_w)
        cfg["constraints"]["long_only"] = bool(long_only)
        cfg["constraints"]["cash_buffer"] = float(cash_buffer)
        CFG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))
        st.success("Saved. Re-run Optimize / Walkforward.")


def load_csv_indexed(path: Path):
    """Backwards-compatible loader used throughout the app."""
    if not Path(path).exists():
        return None
    return load_df(path)

def load_table(path: Path, allow_multiindex: bool = False):
    """Load a general table (index is NOT dates), keep index as-is."""
    if not path.exists():
        return None
    try:
        if allow_multiindex:
            df = pd.read_csv(path, header=[0, 1], index_col=0)
            if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
                df = pd.read_csv(path, index_col=0)
        else:
            df = pd.read_csv(path, index_col=0)
    except Exception:
        df = pd.read_csv(path, index_col=0)
    try:
        df = df.apply(pd.to_numeric, errors="ignore")
    except Exception:
        pass
    return df

def load_ts(path: Path, allow_multiindex: bool = False):
    """Load a time series (index IS dates), parse index to datetime."""
    if not path.exists():
        return None
    if allow_multiindex:
        df = pd.read_csv(path, header=[0, 1], index_col=0)
    else:
        df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index, utc=False, errors="coerce")
    df = df[~df.index.isna()]
    try:
        df = df.apply(pd.to_numeric, errors="ignore")
    except Exception:
        pass
    return df

# ---------- Overview ----------
# ---------- Overview ----------
with tab1:
    col1, col2 = st.columns([2, 1], gap="large")

    with col1:
        st.subheader("Latest Portfolio Weights")
        w = cached_load_df(str(PROC / "latest_weights.csv"))
        if w is not None and not w.empty:
            w_fmt = w.copy()
            for c in w_fmt.columns:
                try:
                    w_fmt[c] = pd.to_numeric(w_fmt[c])
                except Exception:
                    pass
            st.dataframe(w_fmt.style.format("{:.2%}", na_rep="—"))
            st.download_button(
                "Download Weights CSV",
                data=w.to_csv().encode(),
                file_name="latest_weights.csv",
            )
        else:
            st.info("No weights yet. Run the daily pipeline.")

    with col2:
        st.subheader("Latest Metrics")
        m = cached_load_df(str(PROC / "latest_metrics.csv"))
        if m is not None and not m.empty:
            # If latest_metrics.csv was saved as two-column (name,value) or index,value
            if m.shape[1] == 1:
                st.dataframe(m.rename(columns={m.columns[0]: "value"}))
            else:
                st.dataframe(m)
            st.download_button(
                "Download Metrics CSV",
                data=m.to_csv().encode(),
                file_name="latest_metrics.csv",
            )
        else:
            st.info("No metrics yet. Run backtest.")


# ---------- Models ----------
# ---------- Models ----------
with tab2:
    st.subheader("Per-Model KPIs")
    k = cached_load_df(str(PROC / "metrics_by_model.csv"))
    if k is not None and not k.empty:
        k_fmt = k.copy()
        for c in ["CAGR", "Vol", "MaxDD"]:
            if c in k_fmt.columns:
                try:
                    k_fmt[c] = pd.to_numeric(k_fmt[c])
                except Exception:
                    pass
        for c in ["Sharpe", "Sortino"]:
            if c in k_fmt.columns:
                try:
                    k_fmt[c] = pd.to_numeric(k_fmt[c])
                except Exception:
                    pass
        sty = k_fmt.style
        if set(["CAGR", "Vol", "MaxDD"]).issubset(k_fmt.columns):
            sty = sty.format("{:.2%}", subset=["CAGR", "Vol", "MaxDD"], na_rep="—")
        for c in ["Sharpe", "Sortino"]:
            if c in k_fmt.columns:
                sty = sty.format("{:.2f}", subset=[c], na_rep="—")
        st.dataframe(sty)
        st.download_button(
            "Download Per-Model KPIs",
            data=k.to_csv().encode(),
            file_name="metrics_by_model.csv",
        )
    else:
        st.info("Run walk-forward to compute per-model KPIs (see README).")

    st.subheader("Equity Curves")
    eq = cached_load_df(str(PROC / "equity_curves.csv"), ts=True)
    if eq is not None and not eq.empty:
        st.line_chart(eq)
        st.subheader("Drawdowns")
        dd = eq / eq.cummax() - 1.0
        st.line_chart(dd)
        st.caption("Drawdown = Equity / Running Max − 1")
    else:
        st.info("Run walk-forward to build equity curves.")


# ---------- Backtest ----------
with tab3:
    st.subheader("Weights Over Time (Latest Walk-Forward)")

    wt_df = cached_load_df(str(PROC / "weights_timeseries.csv"), ts=True, multi=True)

    if wt_df is None or wt_df.empty:
        st.info("Run walk-forward to generate weights_timeseries.csv.")
    else:
        if isinstance(wt_df.columns, pd.MultiIndex) and wt_df.columns.nlevels == 2:
            models = sorted(set(wt_df.columns.get_level_values(0)))
            model = st.selectbox("Model", models, index=0, key="model_backtest")
            slice_df = wt_df[model].copy()
            assets = st.multiselect(
                "Filter assets", slice_df.columns.tolist(), default=slice_df.columns.tolist()
            )
            plot_df = slice_df[assets] if assets else slice_df
        else:
            # Fallback if CSV is wide without MultiIndex; show all columns
            plot_df = wt_df.copy()
            assets = st.multiselect(
                "Filter assets", plot_df.columns.tolist(), default=plot_df.columns.tolist()
            )
            plot_df = plot_df[assets] if assets else plot_df

        # Format & show
        plot_df_fmt = plot_df.tail(24).copy()
        for c in plot_df_fmt.columns:
            try:
                plot_df_fmt[c] = pd.to_numeric(plot_df_fmt[c])
            except Exception:
                pass

        st.area_chart(plot_df)
        st.caption("Tip: Table shows the last 24 months of per-asset weights.")
        st.dataframe(plot_df_fmt.style.format("{:.2%}", na_rep="—"))
        st.download_button(
            "Download selected model weights",
            data=plot_df.to_csv().encode(),
            file_name=f"weights_timeseries_selected.csv",
        )


# ---------- Stress ----------
# ---------- Stress ----------
with tab4:
    st.subheader("Historical Stress Periods")
    s = cached_load_df(str(PROC / "stress_summary.csv"))
    if s is not None and not s.empty:
        level = st.selectbox("Level", ["asset", "portfolio"], index=0, key="stress_level")
        df = s[s["level"] == level].copy() if "level" in s.columns else s.copy()
        periods_all = sorted(df["period"].unique()) if "period" in df.columns else []
        if periods_all:
            periods = st.multiselect("Periods", periods_all, default=periods_all)
            df = df[df["period"].isin(periods)]
        if set(["name", "period", "return"]).issubset(df.columns):
            pivot = df.pivot(index="name", columns="period", values="return").fillna(0)
            st.dataframe(pivot.style.format("{:+.2%}", na_rep="—"))
        else:
            st.dataframe(df)
        st.download_button(
            "Download filtered stress table",
            data=df.to_csv(index=False).encode(),
            file_name="stress_filtered.csv",
        )
    else:
        st.info("Run `python -m pipeline.stress --config configs/default.yaml` to generate stress summary.")


# ---------- Signals ----------
with tab5:
    st.subheader("Rebalance Recommendations")

    lw = cached_load_df(str(PROC / "latest_weights.csv"))
    prices = cached_load_df(str(ROOT / "data" / "raw" / "prices.csv"), ts=True)
    if lw is None or lw.empty or prices is None or prices.empty:
        st.info("No latest weights yet. Run: optimize → walkforward → backtest.")
    else:
        model = st.selectbox("Model", lw.columns.to_list(), index=0, key="model_signals_ui")

        prevp = PROC / "prev_weights.csv"
        prev_df = cached_load_df(str(prevp))

        col0, colReset = st.columns([4, 1])
        with col0:
            baseline = st.radio(
                "Baseline",
                ["Compare to previous", "Compare to zero"],
                index=0,
                horizontal=True,
                key="baseline_sel",
            )
        with colReset:
            if st.button("Reset previous", help="Set previous weights to zero for all models"):
                (lw * 0.0).to_csv(prevp)
                st.rerun()

        # choose baseline
        if baseline == "Compare to zero" or prev_df is None or prev_df.empty:
            prev_col = pd.Series(0.0, index=lw.index)
        else:
            if model in prev_df.columns:
                prev_col = prev_df[model].reindex(lw.index).fillna(0.0)
            else:
                prev_col = pd.Series(0.0, index=lw.index)

        colA, colB = st.columns(2)
        with colA:
            notional = st.number_input("Notional ($)", value=100000, step=1000, min_value=1000)
        with colB:
            thresh_bps = st.slider("Threshold (bps)", min_value=0, max_value=50, value=10, step=1)

        cur_col = lw[model].reindex(lw.index)
        cur_col = pd.to_numeric(cur_col, errors="coerce").fillna(0.0)
        prev_col = pd.to_numeric(prev_col, errors="coerce").fillna(0.0)

        delta = (cur_col - prev_col).rename("delta_weight").to_frame()
        thresh = thresh_bps / 10000.0
        delta["delta_weight"] = delta["delta_weight"].where(delta["delta_weight"].abs() >= thresh, other=0.0)

        last_px = prices.iloc[-1].reindex(delta.index).ffill()
        dollars = delta["delta_weight"] * notional
        shares = dollars.divide(last_px.replace(0.0, pd.NA))

        st.subheader("Δ Weights")
        st.dataframe(delta.style.format({"delta_weight": "{:+.2%}"}, na_rep="—"))

        st.subheader("Trade Blotter (interactive)")
        blotter = pd.DataFrame({
            "asset": delta.index,
            "delta_weight": delta["delta_weight"].values,
            "dollars": dollars.values,
            "last_price": last_px.values,
            "approx_shares": shares.values,
        }).sort_values("delta_weight")
        st.dataframe(
            blotter.style.format(
                {"delta_weight": "{:+.2%}", "dollars": "{:+,.0f}", "last_price": "{:,.2f}", "approx_shares": "{:+.3f}"},
                na_rep="—"
            )
        )
        st.download_button(
            "Download blotter (interactive)",
            data=blotter.to_csv(index=False).encode(),
            file_name=f"blotter_{model}_{thresh_bps}bps_{int(notional)}.csv",
        )

# ---------- Attribution ----------
with tab6:
    st.subheader("Performance Attribution")

    attr_ret = cached_load_df(str(PROC / "attribution_return.csv"), multi=True)
    attr_risk = cached_load_df(str(PROC / "attribution_risk.csv"), multi=True)

    if (attr_ret is None or attr_ret.empty) and (attr_risk is None or attr_risk.empty):
        st.info(
            "No attribution data found. Generate it with:\n"
            "`python -m pipeline.attribution --config configs/default.yaml`"
        )
    else:
        colL, colR = st.columns(2)

        with colL:
            st.markdown("**Return attribution**")
            if attr_ret is not None and not attr_ret.empty:
                if isinstance(attr_ret.columns, pd.MultiIndex) and attr_ret.columns.nlevels == 2:
                    models_ret = sorted(set(attr_ret.columns.get_level_values(0)))
                    model_ret = st.selectbox("Model (return)", models_ret, key="attr_model_ret")
                    show_ret = attr_ret[model_ret]
                else:
                    show_ret = attr_ret
                st.dataframe(show_ret)
                if not show_ret.empty:
                    last_ret = show_ret.iloc[-1]
                    top_ret = last_ret.sort_values(ascending=False).head(10).to_frame("contrib")
                    st.caption("Top contributors (latest row)")
                    st.bar_chart(top_ret)
                st.download_button(
                    "Download return attribution CSV",
                    data=attr_ret.to_csv().encode(),
                    file_name="attribution_return.csv",
                )
            else:
                st.info("Return attribution file not found.")

        with colR:
            st.markdown("**Risk attribution**")
            if attr_risk is not None and not attr_risk.empty:
                if isinstance(attr_risk.columns, pd.MultiIndex) and attr_risk.columns.nlevels == 2:
                    models_risk = sorted(set(attr_risk.columns.get_level_values(0)))
                    default_idx = models_risk.index(model_ret) if 'model_ret' in locals() and model_ret in models_risk else 0
                    model_risk = st.selectbox("Model (risk)", models_risk, index=default_idx, key="attr_model_risk")
                    show_risk = attr_risk[model_risk]
                else:
                    show_risk = attr_risk
                st.dataframe(show_risk)
                if not show_risk.empty:
                    last_risk = show_risk.iloc[-1]
                    top_risk = last_risk.sort_values(ascending=False).head(10).to_frame("contrib")
                    st.caption("Top contributors (latest row)")
                    st.bar_chart(top_risk)
                st.download_button(
                    "Download risk attribution CSV",
                    data=attr_risk.to_csv().encode(),
                    file_name="attribution_risk.csv",
                )
            else:
                st.info("Risk attribution file not found.")
