# ---- app/dashboard.py (header + sidebar) ----
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# One-time page config
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# --- Project root on path so we can import `utils.*` when running via Streamlit ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# --- API mode toggle & client import (robust) ---
API_BASE = os.getenv("OPTIM_API_URL", "http://127.0.0.1:8000").rstrip("/")
USE_API = bool(os.getenv("OPTIM_API_URL"))
api_get = api_post = None

if USE_API:
    try:
        # preferred location
        from api.api_client import get as api_get, post as api_post  # type: ignore[import]
    except ModuleNotFoundError:
        # try again with explicit root (already added above)
        try:
            from api.api_client import get as api_get, post as api_post  # type: ignore[import]
        except ModuleNotFoundError:
            USE_API = False
            api_get = api_post = None

# --- Local paths (CSV fallback) ---
PROC = ROOT / "data" / "processed"

# ---- Sidebar: background run + live status (app/dashboard.py) ----
import os, time, requests, streamlit as st

API_BASE = os.getenv("OPTIM_API_URL", "http://127.0.0.1:8000").rstrip("/")

with st.sidebar:
    st.markdown("### 🚀 Run Pipeline")
    if st.button("Run All (background)"):
        try:
            r = requests.post(f"{API_BASE}/run/queue", timeout=5)
            r.raise_for_status()
            job_id = r.json().get("job_id")
            st.session_state["job_id"] = job_id
            st.success(f"Queued job: {job_id}")
        except Exception as e:
            st.error(f"Queue failed: {e}")

    # Live status panel
    job_id = st.session_state.get("job_id")
    if job_id:
        try:
            resp = requests.get(f"{API_BASE}/run/status", timeout=5)
            status = resp.json().get("status", {})
        except Exception as e:
            status = {}
            st.error(f"Status fetch failed: {e}")

        state = status.get("state", "—")
        stage = status.get("stage", "—")
        prog = int(status.get("progress", 0))
        st.write(f"**State:** {state} | **Stage:** {stage}")
        st.progress(prog / 100)

        c1, c2 = st.columns(2)
        c1.metric("Started", status.get("started_at", "—"))
        c2.metric("Updated", status.get("updated_at", "—"))

        err = status.get("error")
        if err and state == "error":
            st.error(err)

        # Auto-refresh while running or queued
        if state in {"running", "queued"}:
            time.sleep(2)
            st.rerun()


def api_ok() -> bool:
    if not USE_API:
        return False
    try:
        j = api_get("/health")
        return bool(j.get("ok"))
    except Exception:
        return False

if USE_API:
    st.sidebar.success("API: online ✅" if api_ok() else "API: offline ❌")




# ---------- Cached CSV loader ----------
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
        try:
            df = pd.read_csv(p, header=[0, 1], index_col=index_col)
            if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
                df = pd.read_csv(p, index_col=index_col)
        except Exception:
            df = pd.read_csv(p, index_col=index_col)
    else:
        df = pd.read_csv(p, index_col=index_col)

    if ts:
        try:
            df.index = pd.to_datetime(df.index, format="%Y-%m-%d", errors="coerce")
            if df.index.isna().any():
                df.index = pd.to_datetime(df.index, errors="coerce")
        except Exception:
            df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
    return df

# ---------- Page title & tabs (define ONCE) ----------
st.title("📈 Multi-Asset Portfolio Optimizer (Free Tier)")
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Models", "Backtest", "Stress", "Signals", "Attribution"]
)


# ----- Sidebar: pipeline controls with live status -----
import time, requests, os
import streamlit as st

API_BASE = os.getenv("OPTIM_API_URL", "").rstrip("/")
USE_API = bool(API_BASE)

def poll_status(msg_placeholder):
    if not USE_API:
        return
    for _ in range(300):  # up to ~60s if sleep(0.2); adjust as you like
        try:
            r = requests.get(f"{API_BASE}/run/status", timeout=5)
            j = r.json().get("status", {})
            state = j.get("state", "idle")
            stage = j.get("stage")
            progress = j.get("progress", 0)
            msg_placeholder.write(f"**State:** {state} | **Stage:** {stage} | **Progress:** {progress}%")
            if state in ("done","idle","error"):
                break
        except Exception:
            msg_placeholder.write("Waiting for status…")
        time.sleep(0.2)

st.sidebar.markdown("### ⚙️ Pipeline controls")
col_a, col_b = st.sidebar.columns(2)
with col_a:
    if st.button("Run All", use_container_width=True):
        if USE_API:
            st.toast("Starting full pipeline…", icon="⏳")
            requests.post(f"{API_BASE}/run/all", timeout=600)
            with st.spinner("Running full pipeline…"):
                box = st.empty()
                poll_status(box)
            st.success("Pipeline finished (or stopped).")
            st.rerun()
        else:
            st.warning("Set OPTIM_API_URL to use API mode.")

with col_b:
    if st.button("Refresh Data", use_container_width=True):
        if USE_API:
            st.toast("Fetching data + features…", icon="🔄")
            requests.post(f"{API_BASE}/run/fetch", timeout=5)
            requests.post(f"{API_BASE}/run/features", timeout=5)
            with st.spinner("Refreshing…"):
                box = st.empty()
                poll_status(box)
            st.success("Refresh complete.")
            st.rerun()
        else:
            st.warning("Set OPTIM_API_URL to use API mode.")

with st.sidebar.expander("Run a specific stage"):
    def run_single(path, label):
        if USE_API:
            requests.post(f"{API_BASE}{path}", timeout=5)
            with st.spinner(f"Running {label}…"):
                box = st.empty()
                poll_status(box)
            st.success(f"{label} complete.")
            st.rerun()
        else:
            st.warning("Set OPTIM_API_URL to use API mode.")

    if st.button("Optimize → Weights"):
        run_single("/run/optimize", "Optimize")
    if st.button("Walkforward"):
        run_single("/run/walkforward", "Walkforward")
    if st.button("Backtest"):
        run_single("/run/backtest", "Backtest")
    if st.button("Stress"):
        run_single("/run/stress", "Stress")
    if st.button("Attribution"):
        run_single("/run/attribution", "Attribution")

# ---- Strategy Settings (works in both modes) ----
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
        u = [s.strip() for s in tickers_str.split(",") if s.strip()]
        if USE_API:
            # save to server config
            payload = {
                "universe": u,
                "max_weight": float(max_w),
                "long_only": bool(long_only),
                "cash_buffer": float(cash_buffer),
            }
            api_post("/config", payload)
            st.success("Saved to server config. Re-run Optimize / Walkforward.")
        else:
            # save locally to YAML
            cfg = yaml.safe_load(CFG_PATH.read_text()) if CFG_PATH.exists() else {}
            cfg.setdefault("universe", {})
            cfg["universe"]["equities"] = u
            cfg.setdefault("constraints", {})
            cfg["constraints"]["max_weight"] = float(max_w)
            cfg["constraints"]["long_only"] = bool(long_only)
            cfg["constraints"]["cash_buffer"] = float(cash_buffer)
            CFG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))
            st.success("Saved locally. Re-run Optimize / Walkforward.")

def load_latest_weights_df():
    import pandas as pd
    from pathlib import Path
    if USE_API:
        j = api_get("/weights")
        if j.get("ok") and j.get("weights"):
            rows = j["weights"]
            df = pd.DataFrame(rows).set_index("index")
            return df
        return None
    # fallback: local CSV
    p = Path("data/processed/latest_weights.csv")
    return pd.read_csv(p, index_col=0) if p.exists() else None

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

from pathlib import Path
from datetime import datetime

def fmt_ts(p: Path):
    if not p.exists():
        return "—"
    ts = datetime.fromtimestamp(p.stat().st_mtime)
    return ts.strftime("%Y-%m-%d %H:%M:%S")

PROC = Path("data/processed")

st.markdown("#### 📅 Run Timestamps")
ts_cols = st.columns(3)
with ts_cols[0]:
    st.write("**weights_timeseries.csv**")
    st.write(fmt_ts(PROC / "weights_timeseries.csv"))
with ts_cols[1]:
    st.write("**equity_curves.csv**")
    st.write(fmt_ts(PROC / "equity_curves.csv"))
with ts_cols[2]:
    st.write("**latest_weights.csv**")
    st.write(fmt_ts(PROC / "latest_weights.csv"))

with tab1:
    st.subheader("Latest Portfolio Weights")
    w = cached_load_df(str(PROC / "latest_weights.csv"))
    if w is not None and not w.empty:
        # ensure numeric for pretty formatting (ignore non-numeric safely)
        w_fmt = w.apply(pd.to_numeric, errors="ignore").copy()
        st.dataframe(
            w_fmt.style.format("{:.2%}", na_rep="—"),
            use_container_width=True
        )
        st.download_button(
            "⬇️ Download Weights CSV",
            data=w.to_csv().encode(),
            file_name="latest_weights.csv",
        )
    else:
        st.info("No weights yet. Run the pipeline (Optimize / Walkforward).")

    st.markdown("### Performance summary")
    m = cached_load_df(str(PROC / "latest_metrics.csv"))
    if m is not None and not m.empty:
        # normalize to {metric: value}
        if m.shape[1] == 1:
            m = m.rename(columns={m.columns[0]: "value"})
            metrics = pd.to_numeric(m["value"], errors="coerce").fillna(0.0).to_dict()
        else:
            # assume 2 cols metric,value
            metrics = dict(
                zip(
                    m.iloc[:, 0].astype(str),
                    pd.to_numeric(m.iloc[:, 1], errors="coerce").fillna(0.0),
                )
            )

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("CAGR", f"{metrics.get('CAGR', 0):.2%}")
        col2.metric("Volatility", f"{metrics.get('Vol', 0):.2%}")
        col3.metric("Sharpe", f"{metrics.get('Sharpe', 0):.2f}")
        col4.metric("Sortino", f"{metrics.get('Sortino', 0):.2f}")
        col5.metric("Max Drawdown", f"{metrics.get('MaxDD', 0):.2%}")
        st.download_button(
            "⬇️ Download Metrics CSV",
            data=m.to_csv().encode(),
            file_name="latest_metrics.csv",
        )
    else:
        st.info("No backtest metrics yet. Run Backtest.")

# ---------- Models ----------
with tab2:
    st.subheader("Per-Model KPIs")
    k = cached_load_df(str(PROC / "metrics_by_model.csv"))
    if k is not None and not k.empty:
        k_fmt = k.copy()
        # numeric convert with care
        for c in k.columns:
            k_fmt[c] = pd.to_numeric(k_fmt[c], errors="ignore")
        sty = k_fmt.style
        pct_cols = [c for c in ["CAGR", "Vol", "MaxDD"] if c in k_fmt.columns]
        num_cols = [c for c in ["Sharpe", "Sortino"] if c in k_fmt.columns]
        if pct_cols:
            sty = sty.format("{:.2%}", subset=pct_cols, na_rep="—")
        for c in num_cols:
            sty = sty.format("{:.2f}", subset=[c], na_rep="—")
        st.dataframe(sty, use_container_width=True)
        st.download_button("⬇️ Download KPIs", data=k.to_csv().encode(), file_name="metrics_by_model.csv")
    else:
        st.info("Run Walkforward to compute per-model KPIs.")

    st.markdown("### Equity Curves")
    eq = cached_load_df(str(PROC / "equity_curves.csv"), ts=True)
    if eq is not None and not eq.empty:
        # optional model filter
        models = list(eq.columns)
        chosen = st.multiselect("Select curves to display", models, default=models)
        show = eq[chosen] if chosen else eq
        st.line_chart(show, use_container_width=True)

        st.markdown("#### Drawdowns")
        dd = show / show.cummax() - 1.0
        st.line_chart(dd, use_container_width=True)
        st.caption("Drawdown = Equity / Running Max − 1")

        st.download_button("⬇️ Download Equity Curves", data=eq.to_csv().encode(), file_name="equity_curves.csv")
    else:
        st.info("Run Walkforward to build equity curves.")


# ---------- Backtest ----------
with tab3:
    st.subheader("Weights Over Time (Latest Walk-Forward)")
    wt_df = cached_load_df(str(PROC / "weights_timeseries.csv"), ts=True, multi=True)

    if wt_df is None or wt_df.empty:
        st.info("Run Walkforward to generate weights_timeseries.csv.")
    else:
        if isinstance(wt_df.columns, pd.MultiIndex) and wt_df.columns.nlevels == 2:
            models = sorted(set(wt_df.columns.get_level_values(0)))
            model = st.selectbox("Model", models, index=0, key="model_backtest")
            slice_df = wt_df[model].copy()
            assets = st.multiselect("Filter assets", slice_df.columns.tolist(), default=slice_df.columns.tolist())
            plot_df = slice_df[assets] if assets else slice_df
        else:
            # fallback: no MultiIndex → show all cols
            plot_df = wt_df.copy()
            assets = st.multiselect("Filter assets", plot_df.columns.tolist(), default=plot_df.columns.tolist())
            plot_df = plot_df[assets] if assets else plot_df

        st.area_chart(plot_df, use_container_width=True)
        st.caption("Table shows the last 24 months of per-asset weights.")
        tail_fmt = plot_df.tail(24).apply(pd.to_numeric, errors="ignore")
        st.dataframe(tail_fmt.style.format("{:.2%}", na_rep="—"), use_container_width=True)
        st.download_button("⬇️ Download selected model weights",
                           data=plot_df.to_csv().encode(),
                           file_name=f"weights_timeseries_selected.csv")


# ---------- Stress ----------
# ---------- Stress ----------
with tab4:
    st.subheader("Historical Stress Periods")
    s = cached_load_df(str(PROC / "stress_summary.csv"))
    if s is None or s.empty:
        st.info("Run `python -m pipeline.stress --config configs/default.yaml` to generate stress summary.")
    else:
        level = st.selectbox("Level", ["asset", "portfolio"], index=0, key="stress_level")
        df = s[s["level"] == level].copy() if "level" in s.columns else s.copy()
        periods_all = sorted(df["period"].unique()) if "period" in df.columns else []
        if periods_all:
            periods = st.multiselect("Periods", periods_all, default=periods_all)
            df = df[df["period"].isin(periods)]
        if set(["name", "period", "return"]).issubset(df.columns):
            pivot = df.pivot(index="name", columns="period", values="return").fillna(0)
            st.dataframe(pivot.style.format("{:+.2%}", na_rep="—"), use_container_width=True)
        else:
            st.dataframe(df, use_container_width=True)
        st.download_button("⬇️ Download stress table",
                           data=df.to_csv(index=False).encode(),
                           file_name="stress_filtered.csv")


# ---------- Signals ----------
with tab5:
    st.subheader("Rebalance Recommendations")

    ROOT = Path(__file__).resolve().parents[1]
    lw = cached_load_df(str(PROC / "latest_weights.csv"))
    prices = cached_load_df(str(ROOT / "data" / "raw" / "prices.csv"), ts=True)

    if lw is None or lw.empty or prices is None or prices.empty:
        st.info("No latest weights yet. Run: Optimize → Walkforward → Backtest.")
    else:
        # choose model
        model = st.selectbox("Model", lw.columns.to_list(), index=0, key="model_signals_ui")

        # previous weights (baseline)
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

        if baseline == "Compare to zero" or prev_df is None or prev_df.empty or model not in prev_df.columns:
            prev_col = pd.Series(0.0, index=lw.index)
        else:
            prev_col = prev_df[model].reindex(lw.index).fillna(0.0)

        # interactive controls
        colA, colB, colC = st.columns(3)
        with colA:
            notional = st.number_input("Notional ($)", value=100000, step=1000, min_value=1000)
        with colB:
            thresh_bps = st.slider("Threshold (bps)", min_value=0, max_value=50, value=10, step=1)
        with colC:
            round_lots = st.checkbox("Round lots", value=True)
        lot_size = st.number_input(
            "Lot size (shares)",
            min_value=1,
            value=1,
            step=1,
            help="Applies if 'Round lots' is enabled",
        )

        # compute delta
        cur_col = pd.to_numeric(lw[model], errors="coerce").fillna(0.0)
        prev_col = pd.to_numeric(prev_col, errors="coerce").fillna(0.0)
        delta = (cur_col - prev_col).rename("delta_weight").to_frame()

        # threshold in bps
        thresh = thresh_bps / 10000.0
        delta["delta_weight"] = delta["delta_weight"].where(
            delta["delta_weight"].abs() >= thresh, other=0.0
        )

        # last prices
        last_px = prices.iloc[-1].reindex(delta.index).ffill()

        # dollars & shares
        dollars = (delta["delta_weight"] * notional).rename("dollars")
        shares = dollars.divide(last_px.replace(0.0, pd.NA)).rename("approx_shares")
        if round_lots:
            shares = (shares / lot_size).round() * lot_size

        st.subheader("Δ Weights")
        st.dataframe(
            delta.style.format({"delta_weight": "{:+.2%}"}, na_rep="—"),
            use_container_width=True,
        )

        st.subheader("Trade Blotter")
        blotter = pd.DataFrame(
            {
                "asset": delta.index,
                "delta_weight": delta["delta_weight"].values,
                "dollars": dollars.values,
                "last_price": last_px.values,
                "approx_shares": shares.values,
            }
        ).sort_values("delta_weight")

        st.dataframe(
            blotter.style.format(
                {
                    "delta_weight": "{:+.2%}",
                    "dollars": "{:+,.0f}",
                    "last_price": "{:,.2f}",
                    "approx_shares": "{:+.3f}",
                },
                na_rep="—",
            ),
            use_container_width=True,
        )
        st.download_button(
            "⬇️ Download blotter",
            data=blotter.to_csv(index=False).encode(),
            file_name=f"blotter_{model}_{thresh_bps}bps_{int(notional)}.csv",
        )

        # (Optional) Persist current weights as the new "prev" for next session:
        # lw.to_csv(prevp)
# ---------- Attribution ----------
with tab6:
    st.subheader("Performance Attribution")

    attr_ret = cached_load_df(str(PROC / "attribution_return.csv"), multi=True)
    attr_risk = cached_load_df(str(PROC / "attribution_risk.csv"), multi=True)

    if (attr_ret is None or attr_ret.empty) and (attr_risk is None or attr_risk.empty):
        st.info("No attribution data found. Run: Attribution stage in sidebar.")
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
                st.dataframe(show_ret, use_container_width=True)
                if not show_ret.empty:
                    last_ret = show_ret.iloc[-1]
                    top_ret = last_ret.sort_values(ascending=False).head(10).to_frame("contrib")
                    st.caption("Top contributors (latest row)")
                    st.bar_chart(top_ret)
                st.download_button("⬇️ Download return attribution", data=attr_ret.to_csv().encode(),
                                   file_name="attribution_return.csv")
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
                st.dataframe(show_risk, use_container_width=True)
                if not show_risk.empty:
                    last_risk = show_risk.iloc[-1]
                    top_risk = last_risk.sort_values(ascending=False).head(10).to_frame("contrib")
                    st.caption("Top contributors (latest row)")
                    st.bar_chart(top_risk)
                st.download_button("⬇️ Download risk attribution", data=attr_risk.to_csv().encode(),
                                   file_name="attribution_risk.csv")
            else:
                st.info("Risk attribution file not found.")


if st.sidebar.button("🧹 Clear cache"):
    st.cache_data.clear()
    st.success("Cache cleared")
    st.rerun()

st.sidebar.markdown("### 💾 Saved Strategies")

def api_list_strats():
    j = api_get("/strategies")
    return j.get("items", []) if j.get("ok") else []

def api_get_strat(name: str):
    j = api_get(f"/strategies/{name}")
    return j.get("item") if j.get("ok") else None

def api_save_strat(payload: dict):
    return api_post("/strategies", payload)

def api_delete_strat(name: str):
    import requests
    from urllib.parse import quote
    # simple helper since api_client only has get/post
    BASE = os.getenv("OPTIM_API_URL", "").rstrip("/")
    r = requests.delete(f"{BASE}/strategies/{quote(name)}")
    r.raise_for_status()
    return r.json()

# compose current settings from the form’s last values
current_universe = [s.strip() for s in tickers_str.split(",") if s.strip()]
current_payload = {
    "name": "",  # filled on save
    "universe": current_universe,
    "max_weight": float(max_w),
    "long_only": bool(long_only),
    "cash_buffer": float(cash_buffer),
}

if USE_API:
    col1, col2 = st.sidebar.columns([2,1])
    with col1:
        strat_name = st.text_input("Strategy name", placeholder="e.g., Core_60_40")
    with col2:
        if st.button("Save"):
            if not strat_name:
                st.sidebar.error("Provide a name.")
            else:
                payload = dict(current_payload)
                payload["name"] = strat_name
                api_save_strat(payload)
                st.sidebar.success(f"Saved {strat_name}")

    # list & load/delete
    items = api_list_strats()
    names = [it["name"] for it in items]
    if names:
        selected = st.sidebar.selectbox("Load saved", names)
        cL, cR = st.sidebar.columns(2)
        with cL:
            if st.button("Load"):
                it = api_get_strat(selected)
                if it:
                    # write into local config so the existing form reflects it
                    cfg = {
                        "universe": {"equities": it["universe"]},
                        "constraints": {
                            "max_weight": it["max_weight"],
                            "long_only": it["long_only"],
                            "cash_buffer": it["cash_buffer"],
                        }
                    }
                    # persist locally too so fetch/optimize can read the YAML if needed
                    CFG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))
                    st.success(f"Loaded {selected}. Re-run Optimize/Walkforward.")
        with cR:
            if st.button("Delete"):
                api_delete_strat(selected)
                st.sidebar.warning(f"Deleted {selected}")
else:
    st.sidebar.info("Saved strategies require API mode. Set OPTIM_API_URL and restart.")



if USE_API:
    if st.sidebar.button("Run All for current strategy"):
        api_post("/run/all")  # your server already uses the YAML it manages
        st.sidebar.success("Run submitted. Refresh in a moment.")