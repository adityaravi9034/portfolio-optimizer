# ---------- app/dashboard.py (CLEAN HEADER + SIDEBAR) ----------
import os
import sys
import time
import subprocess
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import yaml

# One-time page config
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
# app/dashboard.py
import os, sys, time, requests
from pathlib import Path
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# --- project root for local imports ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

PROC = ROOT / "data" / "processed"
API_BASE = os.getenv("OPTIM_API_URL", "http://127.0.0.1:8000").rstrip("/")
USE_API = bool(os.getenv("OPTIM_API_URL"))



# Core paths
CFG_PATH = ROOT / "configs" / "default.yaml"
PROC     = ROOT / "data" / "processed"

# API mode toggle
API_BASE = os.getenv("OPTIM_API_URL", "").rstrip("/")
USE_API  = bool(API_BASE)

# Optional API client (only if API mode). If import fails, drop to local mode.
api_get = api_post = None
if USE_API:
    try:
        from api.api_client import get as api_get, post as api_post  # type: ignore[import]
    except Exception:
        USE_API = False
        API_BASE = ""
        api_get = api_post = None

# -------- Local subprocess helpers (used when not in API mode) --------
def _py(*mod_and_args: str) -> list[str]:
    return [sys.executable, "-m", *mod_and_args]

def _run_local(mod: str, *extra_args: str):
    """Run a pipeline module locally (non-API mode)."""
    cmd = _py(mod, "--config", str(CFG_PATH), *extra_args)
    subprocess.check_call(cmd, cwd=str(ROOT))

# ---------- Cached CSV loader ----------
@st.cache_data
def cached_load_df(path: str, *, ts: bool = False, multi: bool = False, index_col: int = 0):
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

# ---------- Sidebar: background run + live status ----------
def _poll_status(panel):
    """Poll API status panel while job runs."""
    for _ in range(300):  # ~60s if sleep(0.2)
        try:
            r = requests.get(f"{API_BASE}/run/status", timeout=5)
            j = r.json().get("status", {})
            state = j.get("state", "idle")
            stage = j.get("stage")
            progress = int(j.get("progress", 0))
            panel.write(f"**State:** {state} | **Stage:** {stage} | **Progress:** {progress}%")
            if state in {"done", "idle", "error"}:
                break
        except Exception:
            panel.write("Waiting for status…")
        time.sleep(0.2)

# ---- Sidebar (single source of truth) ----
import yaml
import os, sys, time
from pathlib import Path
import requests, yaml, streamlit as st
# ---------- Sidebar (single source of truth) ----------
# ─────────────────────────────
# ---------------------------
# Sidebar: controls + settings
# ---------------------------

# ————— helpers —————
def _poll_status_once():
    if not USE_API:
        return {}
    try:
        r = requests.get(f"{API_BASE}/run/status", timeout=5)
        return r.json().get("status", {})
    except Exception:
        return {}

def _post_stage(path: str, label: str, timeout: int = 600):
    if not USE_API:
        st.warning("API mode is off. Set OPTIM_API_URL and restart.")
        return
    with st.spinner(f"Running {label}…"):
        requests.post(f"{API_BASE}{path}", timeout=timeout)
    st.toast(f"{label} stage kicked off", icon="✅")

# ————— UI —————
with st.sidebar:

    # ====== Run pipeline ======
    st.markdown("### 🚀 Run pipeline")

    col_run, col_refresh = st.columns(2)
    with col_run:
        if st.button("Run All", use_container_width=True, key="btn_run_all"):
            if USE_API:
                with st.spinner("Running full pipeline…"):
                    # generous timeout because it runs all stages
                    requests.post(f"{API_BASE}/run/all", timeout=1800)
            else:
                st.warning("API mode is off. Set OPTIM_API_URL and restart.")
    with col_refresh:
        if st.button("Refresh Data", use_container_width=True, key="btn_refresh"):
            if USE_API:
                with st.spinner("Fetching market data + building features…"):
                    requests.post(f"{API_BASE}/run/fetch", timeout=300)
                    requests.post(f"{API_BASE}/run/features", timeout=600)
            else:
                st.warning("API mode is off. Set OPTIM_API_URL and restart.")

    # live status
    status = _poll_status_once() if USE_API else {}
    c1, c2 = st.columns(2)
    c1.metric("Started", status.get("started_at", "—"))
    c2.metric("Updated", status.get("updated_at", "—"))
    st.progress(int(status.get("progress", 0)) / 100 if USE_API else 0.0)
    st.caption(
        f"State: **{status.get('state','idle')}** | Stage: **{status.get('stage','—')}**"
        if USE_API else "API mode is off. Set `OPTIM_API_URL` for background runs."
    )

    # ====== Per-stage controls ======
    with st.expander("Run a specific stage", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Fetch", use_container_width=True, key="stage_fetch"):
                _post_stage("/run/fetch", "Fetch", timeout=300)
            if st.button("Optimize", use_container_width=True, key="stage_opt"):
                _post_stage("/run/optimize", "Optimize", timeout=900)
            if st.button("Stress", use_container_width=True, key="stage_stress"):
                _post_stage("/run/stress", "Stress", timeout=600)
        with c2:
            if st.button("Features", use_container_width=True, key="stage_features"):
                _post_stage("/run/features", "Features", timeout=600)
            if st.button("Walkforward", use_container_width=True, key="stage_wf"):
                _post_stage("/run/walkforward", "Walkforward", timeout=1200)
            if st.button("Backtest", use_container_width=True, key="stage_bt"):
                _post_stage("/run/backtest", "Backtest", timeout=900)

        if st.button("Attribution", use_container_width=True, key="stage_attr"):
            _post_stage("/run/attribution", "Attribution", timeout=600)

    st.markdown("---")

    # ====== Strategy settings (single form, unique key) ======
    st.markdown("### 🧰 Strategy settings")

    # load current YAML to pre-fill
    cfg = {}
    if CFG_PATH.exists():
        try:
            cfg = yaml.safe_load(CFG_PATH.read_text()) or {}
        except Exception:
            cfg = {}
    cur_uni = (cfg.get("universe", {}) or {}).get("equities", [])
    cur_cons = (cfg.get("constraints", {}) or {})
    cur_costs = (cfg.get("costs", {}) or {})

    default_universe = (
        "SPY,QQQ,EFA,EEM,IWM,VNQ,XLK,XLF,XLE,XLV,XLB,XLY,XLP,"
        "IEF,TLT,LQD,HYG,SHY,TIP,BNDX,GLD,SLV,DBC,USO,BTC-USD,ETH-USD"
    )

    # form (single key, no session_state writes to same keys)
    with st.form("strategy_form_main"):
        tickers_text = st.text_area(
            "Universe (comma-separated)",
            value=(",".join(cur_uni) if cur_uni else default_universe),
            height=120,
            key="universe_text_input",
        )
        max_w = st.slider(
            "Max weight per asset",
            0.05, 1.00,
            float(cur_cons.get("max_weight", 0.30)),
            0.05,
            key="max_w_slider",
        )
        long_only = st.checkbox(
            "Long only",
            bool(cur_cons.get("long_only", True)),
            key="long_only_chk",
        )
        cash_buffer = st.slider(
            "Cash buffer",
            0.00, 0.20,
            float(cur_cons.get("cash_buffer", 0.00)),
            0.01,
            key="cash_buf_slider",
        )

        with st.expander("Advanced costs (optional)", expanded=False):
            commission_bps = st.slider(
                "Commission (bps)",
                0.0, 25.0,
                float(cur_costs.get("commission_bps", 1.0)),
                0.5,
                key="commission_bps_slider",
            )
            slippage_bps = st.slider(
                "Slippage (bps)",
                0.0, 25.0,
                float(cur_costs.get("slippage_bps", 1.0)),
                0.5,
                key="slippage_bps_slider",
            )

        submitted_settings = st.form_submit_button("Save Settings")

    if submitted_settings:
        payload = {
            "universe": [s.strip() for s in tickers_text.split(",") if s.strip()],
            "max_weight": float(max_w),
            "long_only": bool(long_only),
            "cash_buffer": float(cash_buffer),
        }
        # costs are optional; include if user touched them
        payload_costs = {
            "commission_bps": float(commission_bps),
            "slippage_bps": float(slippage_bps),
        }

        try:
            if USE_API:
                # update core constraints/universe
                r = requests.post(f"{API_BASE}/config", json=payload, timeout=10)
                r.raise_for_status()
                # merge costs into YAML locally then push? Keep simple: write locally
                cfg.setdefault("costs", {})
                cfg["costs"].update(payload_costs)
                CFG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))
                st.success("Saved via API (core) + YAML (costs). Re-run a stage to apply.")
            else:
                # local YAML only
                cfg.setdefault("universe", {})
                cfg["universe"]["equities"] = payload["universe"]
                cfg.setdefault("constraints", {})
                cfg["constraints"]["max_weight"] = payload["max_weight"]
                cfg["constraints"]["long_only"] = payload["long_only"]
                cfg["constraints"]["cash_buffer"] = payload["cash_buffer"]
                cfg.setdefault("costs", {})
                cfg["costs"].update(payload_costs)
                CFG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))
                st.success("Saved to configs/default.yaml. Run Optimize / Walkforward to refresh.")
        except Exception as e:
            st.error(f"Save failed: {e}")

    # ====== Utilities ======
    st.markdown("---")
    if st.button("🧹 Clear cache", use_container_width=True, key="btn_clear_cache"):
        st.cache_data.clear()
        st.success("Streamlit cache cleared.")

    # ====== Saved Strategies (optional API) ======
    st.markdown("### 💾 Saved Strategies")
    name = st.text_input("Strategy name", key="saved_name")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("Save", use_container_width=True, key="btn_save_strategy"):
            if not name:
                st.warning("Enter a name.")
            else:
                try:
                    if USE_API:
                        requests.post(
                            f"{API_BASE}/strategies",
                            json={
                                "name": name,
                                "universe": [s.strip() for s in st.session_state.get("universe_text_input","").split(",") if s.strip()],
                                "max_weight": float(st.session_state.get("max_w_slider", 0.3)),
                                "long_only": bool(st.session_state.get("long_only_chk", True)),
                                "cash_buffer": float(st.session_state.get("cash_buf_slider", 0.0)),
                            },
                            timeout=10,
                        ).raise_for_status()
                        st.success(f"Saved '{name}' to API.")
                    else:
                        st.info("API mode off; not persisted to DB. (YAML already saved above.)")
                except Exception as e:
                    st.error(f"Save failed: {e}")
    with col_s2:
        if USE_API and st.button("Load", use_container_width=True, key="btn_load_strategy"):
            try:
                r = requests.get(f"{API_BASE}/strategies/{name}", timeout=5)
                r.raise_for_status()
                item = r.json().get("item", {})
                # update YAML locally so rest of app sees it
                cfg.setdefault("universe", {})
                cfg["universe"]["equities"] = item.get("universe", [])
                cfg.setdefault("constraints", {})
                cfg["constraints"]["max_weight"] = float(item.get("max_weight", 0.3))
                cfg["constraints"]["long_only"] = bool(item.get("long_only", True))
                cfg["constraints"]["cash_buffer"] = float(item.get("cash_buffer", 0.0))
                CFG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))
                st.success(f"Loaded '{name}'. Re-open sidebar to see new defaults.")
            except Exception as e:
                st.error(f"Load failed: {e}")

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

# ---------- Page title & tabs ----------
st.title("📈 Multi-Asset Portfolio Optimizer (Free Tier)")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Overview", "Models", "Backtest", "Stress", "Signals", "Attribution"]
)

# ----- helpers for loading & metrics -----
from math import sqrt

TRADING_DAYS = 252

def _safe_read_csv(path: Path, **kwargs):
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path, **kwargs)

def _compute_metrics_from_eq(eq: pd.Series) -> dict:
    """eq is an equity curve indexed by date, starting ~1.0"""
    if eq is None or eq.empty or len(eq) < 2:
        return {"CAGR": 0.0, "Vol": 0.0, "Sharpe": 0.0, "Sortino": 0.0, "MaxDD": 0.0}
    r = np.log(eq).diff().dropna()
    cagr = float(eq.iloc[-1] ** (TRADING_DAYS / len(eq)) - 1.0)
    vol  = float(np.sqrt(TRADING_DAYS) * r.std(ddof=0))
    downside = r[r < 0]
    sortino = float((r.mean() * TRADING_DAYS) / (downside.std(ddof=0) * np.sqrt(TRADING_DAYS))) if not downside.empty and downside.std(ddof=0) > 0 else 0.0
    sharpe = float((r.mean() * TRADING_DAYS) / vol) if vol > 0 else 0.0
    # max drawdown
    roll_max = eq.cummax()
    dd = (eq / roll_max) - 1.0
    maxdd = float(dd.min()) if not dd.empty else 0.0
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino, "MaxDD": maxdd}

def _fmt_pct(x): 
    try: return f"{x:.2%}"
    except: return "—"

with tab1:
    st.subheader("Latest Portfolio Weights")

    w_path  = PROC / "latest_weights.csv"
    kpi_path = PROC / "metrics_by_model.csv"
    eq_path = PROC / "equity_curves.csv"

    w = _safe_read_csv(w_path, index_col=0)
    if w is not None:
        # model select for the summary
        models = list(w.columns)
        sel_model = st.selectbox("Model for summary", models, index=0, key="model_overview")

        w_fmt = w.copy()
        for c in w_fmt.columns:
            w_fmt[c] = pd.to_numeric(w_fmt[c], errors="coerce")
        st.dataframe(w_fmt.style.format("{:.2%}"), use_container_width=True)
        st.download_button("Download Weights CSV", data=w.to_csv().encode(), file_name="latest_weights.csv")
    else:
        st.info("No weights yet. Run the pipeline.")
        sel_model = None

    st.markdown("### Performance summary")

    # Prefer metrics_by_model; fall back to computing from equity curve
    metrics = None
    kdf = _safe_read_csv(kpi_path, index_col=0)
    if sel_model and kdf is not None and sel_model in kdf.index:
        row = kdf.loc[sel_model]
        metrics = {
            "CAGR":  float(pd.to_numeric(row.get("CAGR", 0), errors="coerce") or 0),
            "Vol":   float(pd.to_numeric(row.get("Vol", 0), errors="coerce") or 0),
            "Sharpe":float(pd.to_numeric(row.get("Sharpe", 0), errors="coerce") or 0),
            "Sortino":float(pd.to_numeric(row.get("Sortino", 0), errors="coerce") or 0),
            "MaxDD": float(pd.to_numeric(row.get("MaxDD", 0), errors="coerce") or 0),
        }
    elif sel_model:
        # compute from equity curve
        eq = _safe_read_csv(eq_path, index_col=0, parse_dates=True)
        if eq is not None and sel_model in eq.columns:
            metrics = _compute_metrics_from_eq(eq[sel_model])

    if metrics is None:
        metrics = {"CAGR":0.0,"Vol":0.0,"Sharpe":0.0,"Sortino":0.0,"MaxDD":0.0}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR",    _fmt_pct(metrics["CAGR"]))
    c2.metric("Volatility", _fmt_pct(metrics["Vol"]))
    c3.metric("Sharpe",  f"{metrics['Sharpe']:.2f}")
    c4.metric("Sortino", f"{metrics['Sortino']:.2f}")
    c5.metric("Max Drawdown", _fmt_pct(metrics["MaxDD"]))

    if kdf is not None:
        st.download_button("Download Metrics CSV", data=kdf.to_csv().encode(), file_name="metrics_by_model.csv")

# ---------- Models (Per-Model KPIs + Curves) ----------
with tab2:
    st.subheader("Per-Model KPIs")
    k = cached_load_df(str(PROC / "metrics_by_model.csv"))
    if k is not None and not k.empty:
        # Safely coerce to numeric (no FutureWarning)
        k_fmt = k.copy()
        for c in k_fmt.columns:
            k_fmt[c] = pd.to_numeric(k_fmt[c], errors="coerce")

        pct_cols = [c for c in ["CAGR", "Vol", "MaxDD"] if c in k_fmt.columns]
        num_cols = [c for c in ["Sharpe", "Sortino"] if c in k_fmt.columns]

        sty = k_fmt.style
        if pct_cols:
            sty = sty.format("{:.2%}", subset=pct_cols, na_rep="—")
        if num_cols:
            for c in num_cols:
                sty = sty.format("{:.2f}", subset=[c], na_rep="—")

        st.dataframe(sty, use_container_width=True)
        st.download_button("⬇️ Download KPIs",
                           data=k.to_csv().encode(),
                           file_name="metrics_by_model.csv")
    else:
        st.info("Run Walkforward to compute per-model KPIs.")

    st.markdown("### Equity Curves")
    eq = cached_load_df(str(PROC / "equity_curves.csv"), ts=True)
    if eq is not None and not eq.empty:
        models = list(eq.columns)
        chosen = st.multiselect("Select curves to display", models, default=models)
        show = eq[chosen] if chosen else eq

        st.line_chart(show, use_container_width=True)

        st.markdown("#### Drawdowns")
        dd = show / show.cummax() - 1.0
        st.line_chart(dd, use_container_width=True)
        st.caption("Drawdown = Equity / Running Max − 1")

        st.download_button("⬇️ Download Equity Curves",
                           data=eq.to_csv().encode(),
                           file_name="equity_curves.csv")
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
            assets = st.multiselect("Filter assets",
                                    slice_df.columns.tolist(),
                                    default=slice_df.columns.tolist())
            plot_df = slice_df[assets] if assets else slice_df
        else:
            # Fallback: no MultiIndex → show all cols
            plot_df = wt_df.copy()
            assets = st.multiselect("Filter assets",
                                    plot_df.columns.tolist(),
                                    default=plot_df.columns.tolist())
            plot_df = plot_df[assets] if assets else plot_df

        st.area_chart(plot_df, use_container_width=True)
        st.caption("Table shows the last 24 months of per-asset weights.")

        # Safely coerce to numeric for formatting (no FutureWarning)
        tail = plot_df.tail(24).copy()
        for c in tail.columns:
            tail[c] = pd.to_numeric(tail[c], errors="coerce")

        st.dataframe(tail.style.format("{:.2%}", na_rep="—"),
                     use_container_width=True)

        st.download_button("⬇️ Download selected model weights",
                           data=plot_df.to_csv().encode(),
                           file_name="weights_timeseries_selected.csv")


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
current_universe = [s.strip() for s in tickers_text.split(",") if s.strip()]
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

