# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
# app/dashboard.py
# ---------- Streamlit page config MUST be first ----------
import streamlit as st
st.set_page_config(page_title="Portfolio Optimizer", layout="wide")

# ---------- Put repo root on sys.path so `from app...` works ----------
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------- Standard libs ----------
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yaml
from dotenv import load_dotenv  # pip install python-dotenv

# ---------- UI components (ABSOLUTE imports; no relative dots) ----------
from app.components.theme import apply_compact_theme
from app.components.presets import render_presets_panel
from app.components.live_tiles import render_live_kpis
from app.components.controls import render_strategy_controls
from app.components.charts import plot_equity_curves, plot_weights_area
from app.components.reports import render_reports_panel
from app.components.universe import render_universe_editor
from app.components.versions import render_versions_panel
# at the top with other components
from app.components.frontier import render_frontier
from app.components.builder import render_dashboard_builder
from app.components.trading import trading_panel
from app.components.ml import render_ml_panel


# ---------- Theme ----------
apply_compact_theme()

# tiny build banner (optional)
st.caption(f"Build: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • File: {__file__} • PID: {os.getpid()}")

# ---------- Paths & API flags ----------
CFG_PATH = ROOT / "configs" / "default.yaml"
PROC     = ROOT / "data" / "processed"

load_dotenv()  # read .env if present
API_BASE = (os.getenv("OPTIM_API_URL") or "").strip().rstrip("/")
USE_API  = bool(API_BASE)
_api_import_error = None  # set ahead of use

st.session_state["api_base"] = API_BASE

# Optional API client import (for API-mode buttons)
if USE_API:
    try:
        from api.api_client import get as api_get, post as api_post  # type: ignore[import]
    except Exception as e:
        _api_import_error = e
        USE_API = False

# ---------- Helpers ----------
def _py(*mod_and_args: str) -> list[str]:
    return [sys.executable, "-m", *mod_and_args]

def _run_local(mod: str, *extra_args: str):
    """Run a pipeline module locally (non-API mode)."""
    cmd = _py(mod, "--config", str(CFG_PATH), *extra_args)
    subprocess.check_call(cmd, cwd=str(ROOT))


@st.cache_data
def cached_load_df(
    path: str,
    *,
    ts: bool = False,
    multi: bool = False,
    index_col: int = 0,
    _cache_bust: tuple[int, int] | None = None,  # (mtime, size) for cache key
):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None

    # ----- cache key input -----
    mtime = int(p.stat().st_mtime)
    size  = int(p.stat().st_size)
    if _cache_bust is None:
        _cache_bust = (mtime, size)

    # ----- load -----
    if multi:
        try:
            df = pd.read_csv(p, header=[0, 1], index_col=index_col)
            if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels != 2:
                df = pd.read_csv(p, index_col=index_col)
        except Exception:
            df = pd.read_csv(p, index_col=index_col)
    else:
        df = pd.read_csv(p, index_col=index_col)

    # timestamps if requested
    if ts:
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]

    # gently coerce numerics
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def _safe_read_csv(path: Path, **kwargs):
    if not path.exists() or path.stat().st_size == 0:
        return None
    return pd.read_csv(path, **kwargs)

def fmt_ts(p: Path) -> str:
    try:
        if not p.exists():
            return "—"
        ts = datetime.fromtimestamp(p.stat().st_mtime)
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return "—"

TRADING_DAYS = 252
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
    roll_max = eq.cummax()
    dd = (eq / roll_max) - 1.0
    maxdd = float(dd.min()) if not dd.empty else 0.0
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "Sortino": sortino, "MaxDD": maxdd}

def _fmt_pct(x):
    try:
        return f"{x:.2%}"
    except Exception:
        return "—"

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

# Pre-compute cache bust keys (safe if files don't exist)
eq_path = PROC / "equity_curves.csv"
wt_path = PROC / "weights_timeseries.csv"
eq = cached_load_df(
    str(eq_path),
    ts=True,
    _cache_bust=(int(eq_path.stat().st_mtime), int(eq_path.stat().st_size)) if eq_path.exists() else (0, 0),
)
wt_df = cached_load_df(
    str(wt_path),
    ts=True, multi=True,
    _cache_bust=(int(wt_path.stat().st_mtime), int(wt_path.stat().st_size)) if wt_path.exists() else (0, 0),
)




# =========================
# Sidebar: pipeline + settings + saved strategies + health + auto-refresh
# =========================
import yaml
import requests
import time
from urllib.parse import quote
from pathlib import Path

# prerequisites assumed set earlier in the file:
# USE_API: bool
# API_BASE: str
# CFG_PATH = Path("configs/default.yaml")

def _poll_status_once() -> dict:
    try:
        r = requests.get(f"{API_BASE}/run/status", timeout=5)
        r.raise_for_status()
        return r.json().get("status", {}) or {}
    except Exception:
        return {}

def _post_stage(path: str, label: str, timeout: int = 600) -> None:
    if not USE_API:
        st.warning("API mode is off. Set OPTIM_API_URL and restart.")
        return
    try:
        with st.spinner(f"Running {label}…"):
            r = requests.post(f"{API_BASE}{path}", timeout=timeout)
            r.raise_for_status()
            st.success(f"{label} complete")
    except Exception as e:
        st.error(f"{label} failed: {e}")

def render_presets_panel(api_base: str, use_api: bool):
    st.markdown("### 📦 Presets")
    c1, c2 = st.columns([2,1])
    with c1:
        preset = st.selectbox(
            "Choose a preset",
            ["Default (26 assets)", "Defensive Tilt", "Aggressive Tilt", "Crypto-Lite"],
            index=0
        )
    autorun = c2.checkbox("Run after apply", value=False, help="Queue full pipeline after saving preset")

    if st.button("Apply preset", use_container_width=True, key="btn_apply_preset"):
        try:
            # simple presets writing into YAML
            cfg = yaml.safe_load(CFG_PATH.read_text()) or {}
            cfg.setdefault("universe", {})
            cfg.setdefault("constraints", {})
            if preset == "Default (26 assets)":
                # leave as-is (your default.yaml already contains the universe)
                pass
            elif preset == "Defensive Tilt":
                cfg["constraints"]["max_weight"] = 0.20
                cfg["constraints"]["cash_buffer"] = 0.05
            elif preset == "Aggressive Tilt":
                cfg["constraints"]["max_weight"] = 0.40
                cfg["constraints"]["cash_buffer"] = 0.00
            elif preset == "Crypto-Lite":
                eq = (cfg.get("universe", {}).get("equities") or [])
                # keep crypto light by removing BTC/ETH if present
                cfg["universe"]["equities"] = [t for t in eq if t not in ("BTC-USD","ETH-USD")]

            CFG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))
            st.success(f"Applied preset: {preset}")
            if use_api and autorun:
                r = requests.post(f"{api_base}/run/queue", timeout=5)
                r.raise_for_status()
                job_id = r.json().get("job_id")
                st.session_state["job_id"] = job_id
                st.info(f"Queued pipeline (job {job_id})")
                st.rerun()
        except Exception as e:
            st.error(f"Preset apply failed: {e}")
    return st.session_state.get("job_id")

with st.sidebar:
    st.markdown("### 🚀 Run pipeline")

    # live status (for button disabling)
    status = _poll_status_once() if USE_API else {}
    running = status.get("state") in {"running", "queued"} if USE_API else False

    col_run, col_refresh = st.columns(2)
    with col_run:
        if st.button("Run All", use_container_width=True, key="btn_run_all", disabled=(not USE_API) or running):
            if USE_API:
                try:
                    with st.spinner("Queuing full pipeline…"):
                        r = requests.post(f"{API_BASE}/run/queue", timeout=5)
                        r.raise_for_status()
                        st.session_state["job_id"] = r.json().get("job_id")
                        st.success("Queued!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Queue failed: {e}")
            else:
                st.warning("API mode is off. Set OPTIM_API_URL and restart.")
    with col_refresh:
        if st.button("Refresh Data", use_container_width=True, key="btn_refresh", disabled=(not USE_API) or running):
            if USE_API:
                # simplest: queue full run as well, or add a /run/partial on the API
                try:
                    with st.spinner("Queuing fetch+features…"):
                        r = requests.post(f"{API_BASE}/run/queue", timeout=5)
                        r.raise_for_status()
                        st.session_state["job_id"] = r.json().get("job_id")
                        st.success("Queued!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Queue failed: {e}")
            else:
                st.warning("API mode is off. Set OPTIM_API_URL and restart.")

    # --- Presets with optional auto-run ---
    job_from_preset = render_presets_panel(api_base=API_BASE, use_api=USE_API and (not running))
    if job_from_preset:
        st.session_state["job_id"] = job_from_preset

    # --- Live status + progress ---
    st.markdown("### 📊 Run status")
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
            if st.button("Fetch", use_container_width=True, key="stage_fetch", disabled=(not USE_API) or running):
                _post_stage("/run/fetch", "Fetch", timeout=300)
            if st.button("Optimize", use_container_width=True, key="stage_opt", disabled=(not USE_API) or running):
                _post_stage("/run/optimize", "Optimize", timeout=900)
            if st.button("Stress", use_container_width=True, key="stage_stress", disabled=(not USE_API) or running):
                _post_stage("/run/stress", "Stress", timeout=600)
        with c2:
            if st.button("Features", use_container_width=True, key="stage_features", disabled=(not USE_API) or running):
                _post_stage("/run/features", "Features", timeout=600)
            if st.button("Walkforward", use_container_width=True, key="stage_wf", disabled=(not USE_API) or running):
                _post_stage("/run/walkforward", "Walkforward", timeout=1200)
            if st.button("Backtest", use_container_width=True, key="stage_bt", disabled=(not USE_API) or running):
                _post_stage("/run/backtest", "Backtest", timeout=900)

        if st.button("Attribution", use_container_width=True, key="stage_attr", disabled=(not USE_API) or running):
            _post_stage("/run/attribution", "Attribution", timeout=600)

    st.markdown("---")

    # ====== Strategy settings ======
    st.markdown("### Strategy settings")
    render_strategy_controls()  # your existing component

    # ====== Saved strategies (API mode only) ======
    with st.expander("💾 Saved strategies", expanded=False):
        if USE_API:
            # Load list
            items = []
            try:
                r = requests.get(f"{API_BASE}/strategies", timeout=10)
                r.raise_for_status()
                items = r.json().get("items", [])
            except Exception as e:
                st.error(f"Load failed: {e}")

            # Save current YAML
            new_name = st.text_input("Name", placeholder="e.g., Core_60_40", key="strat_name")
            if st.button("Save current as strategy", use_container_width=True, key="btn_save_strat"):
                try:
                    cfg = yaml.safe_load(CFG_PATH.read_text()) or {}
                    payload = {
                        "name": new_name.strip(),
                        "universe": (cfg.get("universe", {}) or {}).get("equities", []),
                        "max_weight": float((cfg.get("constraints", {}) or {}).get("max_weight", 0.30)),
                        "long_only": bool((cfg.get("constraints", {}) or {}).get("long_only", True)),
                        "cash_buffer": float((cfg.get("constraints", {}) or {}).get("cash_buffer", 0.00)),
                    }
                    if not payload["name"]:
                        st.warning("Enter a name.")
                    else:
                        rr = requests.post(f"{API_BASE}/strategies", json=payload, timeout=10)
                        rr.raise_for_status()
                        st.success(f"Saved {payload['name']}")
                        st.rerun()
                except Exception as e:
                    st.error(f"Save failed: {e}")

            # Load / delete
            names = [it["name"] for it in items]
            if names:
                pick = st.selectbox("Load / delete", names, key="pick_saved_strat")
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Load", use_container_width=True, key="btn_load_strat"):
                        try:
                            rr = requests.get(f"{API_BASE}/strategies/{quote(pick)}", timeout=10)
                            rr.raise_for_status()
                            it = rr.json().get("item")
                            if it:
                                cfg = yaml.safe_load(CFG_PATH.read_text()) or {}
                                cfg.setdefault("universe", {})["equities"] = it["universe"]
                                cfg.setdefault("constraints", {}).update({
                                    "max_weight": it["max_weight"],
                                    "long_only": it["long_only"],
                                    "cash_buffer": it["cash_buffer"],
                                })
                                CFG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))
                                st.success(f"Loaded {pick} into YAML. Re-run stages to apply.")
                        except Exception as e:
                            st.error(f"Load failed: {e}")
                with c2:
                    if st.button("Delete", use_container_width=True, key="btn_delete_strat"):
                        try:
                            rr = requests.delete(f"{API_BASE}/strategies/{quote(pick)}", timeout=10)
                            rr.raise_for_status()
                            st.warning(f"Deleted {pick}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Delete failed: {e}")
        else:
            st.info("Enable API mode (set OPTIM_API_URL) to save/share strategies.")
    if st.button("Run ML", use_container_width=True, key="btn_run_ml", disabled=(not USE_API) or running):
        try:
            with st.spinner("Training ensemble & forecasting…"):
                r = requests.post(f"{API_BASE}/run/ml", timeout=600)
            st.success("ML complete")
        except Exception as e:
            st.error(f"ML failed: {e}")

    st.markdown("---")

    

    # ====== Clear cache ======
    if st.button("🧹 Clear cache", use_container_width=True, key="btn_clear_cache_sidebar_v2"):
        st.cache_data.clear()
        st.toast("Cache cleared", icon="🧹")
        st.rerun()

    # ====== API health + auto-refresh ======
    if USE_API:
        try:
            ok = requests.get(f"{API_BASE}/health", timeout=3).ok
            st.success("API online ✅" if ok else "API offline ❌")
        except Exception as e:
            st.error(f"API check failed: {e}")
    else:
        st.info("API mode is off. Set OPTIM_API_URL to enable backend calls.")

    try:
        from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh
        st.markdown("### 🔄 Live refresh")
        enable_auto = st.checkbox("Auto-refresh", value=True, help="Re-run the page on a timer")
        interval = st.slider("Every (seconds)", 5, 60, 10, help="UI updates, KPIs & charts refresh")
        if enable_auto and (status.get("state") in {"running", "queued"} if USE_API else False):
            st_autorefresh(interval=interval * 1000, limit=0, key="auto_refresh")
    except Exception:
        pass

    st.caption(f"API_BASE: {API_BASE or '—'} | USE_API={USE_API}")

    

# =========================
# Main body
# =========================
st.title("📈 Multi-Asset Portfolio Optimizer")

# Phase 1 live KPIs (top strip)
st.subheader("Live KPIs")
render_live_kpis()
st.divider()

# Quick run timestamps row
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

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11,tab12 = st.tabs(
    ["Overview", "Models", "Backtest", "Stress", "Signals", "Attribution", "Collab", "Universe","Versions","Frontier","Builder","ML"]
)


# ---------- Overview ----------
with tab1:
    st.subheader("Latest Portfolio Weights")

    w_path  = PROC / "latest_weights.csv"
    kpi_path = PROC / "metrics_by_model.csv"
    eqp = PROC / "equity_curves.csv"

    w = _safe_read_csv(w_path, index_col=0)
    if w is not None:
        models = list(w.columns)
        sel_model = st.selectbox("Model for summary", models, index=0, key="model_overview")

        w_fmt = w.copy()
        for c in w_fmt.columns:
            w_fmt[c] = pd.to_numeric(w_fmt[c], errors="coerce")
        st.dataframe(w_fmt.style.format("{:.2%}", na_rep="—"), use_container_width=True)
        st.download_button("⬇️ Download Weights", data=w.to_csv().encode(), file_name="latest_weights.csv")
    else:
        st.info("No weights yet. Run the pipeline.")
        sel_model = None

    st.markdown("### Performance summary")
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
        eq_local = _safe_read_csv(eqp, index_col=0, parse_dates=True)
        if eq_local is not None and sel_model in eq_local.columns:
            metrics = _compute_metrics_from_eq(eq_local[sel_model])

    if metrics is None:
        metrics = {"CAGR":0.0,"Vol":0.0,"Sharpe":0.0,"Sortino":0.0,"MaxDD":0.0}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR",       _fmt_pct(metrics["CAGR"]))
    c2.metric("Volatility", _fmt_pct(metrics["Vol"]))
    c3.metric("Sharpe",     f"{metrics['Sharpe']:.2f}")
    c4.metric("Sortino",    f"{metrics['Sortino']:.2f}")
    c5.metric("Max DD",     _fmt_pct(metrics["MaxDD"]))

    if kdf is not None:
        st.download_button("⬇️ Download Metrics", data=kdf.to_csv().encode(), file_name="metrics_by_model.csv")
    
    from app.components.strategies import render_strategy_manager

# ... later in the page (e.g., in the Overview tab under Settings) ...
    st.markdown("### Strategy presets")
    render_strategy_manager()



# ---------- Models (ONLY here) ----------
with tab2:
    st.subheader("Models")

    view = st.selectbox(
        "Select view",
        [
            "Per-Model KPIs (table)",
            "Equity Curves (interactive)",
            "Weights Over Time (stacked)",
            "Drawdowns only",
        ],
        index=0,
        key="models_view_selector",
    )

    if view == "Per-Model KPIs (table)":
        k = cached_load_df(str(PROC / "metrics_by_model.csv"))
        if k is not None and not k.empty:
            k_fmt = k.copy()
            for c in k_fmt.columns:
                k_fmt[c] = pd.to_numeric(k_fmt[c], errors="coerce")
            pct_cols = [c for c in ["CAGR", "Vol", "MaxDD"] if c in k_fmt.columns]
            num_cols = [c for c in ["Sharpe", "Sortino"] if c in k_fmt.columns]
            sty = k_fmt.style
            if pct_cols:
                sty = sty.format("{:.2%}", subset=pct_cols, na_rep="—")
            if num_cols:
                sty = sty.format("{:.2f}", subset=num_cols, na_rep="—")
            st.dataframe(sty, use_container_width=True)
            st.download_button("⬇️ Download KPIs", data=k.to_csv().encode(), file_name="metrics_by_model.csv")
        else:
            st.info("Run Walkforward to compute per-model KPIs.")

    elif view == "Equity Curves (interactive)":
        plot_equity_curves()

    elif view == "Weights Over Time (stacked)":
        plot_weights_area(key_prefix="weights_main")

    elif view == "Drawdowns only":
        eq_local = cached_load_df(str(PROC / "equity_curves.csv"), ts=True)
        if eq_local is not None and not eq_local.empty:
            models = list(eq_local.columns)
            chosen = st.multiselect("Curves", models, default=models, key="eq_dd_only_models")
            show = eq_local[chosen] if chosen else eq_local
            dd = show / show.cummax() - 1.0
            st.line_chart(dd, use_container_width=True)
            st.caption("Drawdown = Equity / Running Max − 1")
            st.download_button(
                "⬇️ Download Equity Curves",
                data=eq_local.to_csv().encode(),
                file_name="equity_curves.csv",
                key="dl_eq_curves_dashboard_main",
            )
        else:
            st.info("Run Walkforward to build equity curves.")

# (Other tabs left as placeholders for your existing Backtest/Stress/Signals/Attribution content)

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
            plot_df = wt_df.copy()
            assets = st.multiselect("Filter assets", plot_df.columns.tolist(), default=plot_df.columns.tolist())
            plot_df = plot_df[assets] if assets else plot_df

        st.area_chart(plot_df, use_container_width=True)
        st.caption("Table shows the last 24 months of per-asset weights.")

        tail = plot_df.tail(24).copy()
        for c in tail.columns:
            tail[c] = pd.to_numeric(tail[c], errors="coerce")
        st.dataframe(tail.style.format("{:.2%}", na_rep="—"), use_container_width=True)

        st.download_button("⬇️ Download selected model weights",
                           data=plot_df.to_csv().encode(),
                           file_name="weights_timeseries_selected.csv")

# ---------- Stress ----------
# ---- add near your helpers (reuse your cached loader but don't set index) ----
@st.cache_data
def load_stress_csv(path: str, _cache_bust: tuple[int, int]):
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    # IMPORTANT: no index_col here; new stress.csv is saved without index
    df = pd.read_csv(p)
    # normalize columns just in case
    wanted = ["period","start","end","level","name","total_return","max_drawdown"]
    df = df[[c for c in wanted if c in df.columns]]
    return df

# ---- in the Stress tab content, replace the existing block with: ----
with tab4:  # or whatever tab your "Stress" is
    st.subheader("Historical Stress Periods")

    stress_path = PROC / "stress_periods.csv"
    stress_df = None
    if stress_path.exists() and stress_path.stat().st_size > 0:
        stress_df = load_stress_csv(
            str(stress_path),
            _cache_bust=(int(stress_path.stat().st_mtime), int(stress_path.stat().st_size))
        )

    if stress_df is None or stress_df.empty:
        st.info("No stress results found. Run the Stress stage.")
    else:
        level = st.selectbox("Level", ["portfolio", "asset"], index=0, key="stress_level")
        df_show = stress_df[stress_df["level"] == level].copy()

        # nice formatting
        if "total_return" in df_show:   df_show["total_return"]  = pd.to_numeric(df_show["total_return"],  errors="coerce")
        if "max_drawdown" in df_show:   df_show["max_drawdown"]  = pd.to_numeric(df_show["max_drawdown"], errors="coerce")
        fmt = df_show.style.format({
            "total_return": "{:.2%}",
            "max_drawdown": "{:.2%}",
        }, na_rep="—")

        st.dataframe(fmt, use_container_width=True, height=360)
        st.download_button(
            "⬇️ Download stress table",
            data=df_show.to_csv(index=False).encode(),
            file_name=f"stress_periods_{level}.csv",
            key="dl_stress_table"
        )

        
# ---------- Signals ----------
# ---------- Signals ----------
with tab5:
    st.subheader("Rebalance Recommendations")

    lw = cached_load_df(str(PROC / "latest_weights.csv"))
    prices = cached_load_df(str(ROOT / "data" / "raw" / "prices.csv"), ts=True)

    if lw is None or lw.empty or prices is None or prices.empty:
        st.info("No latest weights yet. Run: Optimize → Walkforward → Backtest.")
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
            if st.button(
                "Reset previous",
                help="Set previous weights to zero for all models",
                key="signals_reset_prev",
            ):
                (lw * 0.0).to_csv(prevp)
                st.rerun()

        if baseline == "Compare to zero" or prev_df is None or prev_df.empty or model not in prev_df.columns:
            prev_col = pd.Series(0.0, index=lw.index)
        else:
            prev_col = prev_df[model].reindex(lw.index).fillna(0.0)

        colA, colB, colC = st.columns(3)
        with colA:
            notional = st.number_input("Notional ($)", value=100000, step=1000, min_value=1000, key="signals_notional")
        with colB:
            thresh_bps = st.slider("Threshold (bps)", min_value=0, max_value=50, value=10, step=1, key="signals_thresh_bps")
        with colC:
            round_lots = st.checkbox("Round lots", value=True, key="signals_round_lots")
        lot_size = st.number_input(
            "Lot size (shares)",
            min_value=1, value=1, step=1,
            help="Applies if 'Round lots' is enabled",
            key="signals_lot_size",
        )

        cur_col = pd.to_numeric(lw[model], errors="coerce").fillna(0.0)
        prev_col = pd.to_numeric(prev_col, errors="coerce").fillna(0.0)
        delta = (cur_col - prev_col).rename("delta_weight").to_frame()

        thresh = thresh_bps / 10000.0
        delta["delta_weight"] = delta["delta_weight"].where(delta["delta_weight"].abs() >= thresh, other=0.0)

        last_px = prices.iloc[-1].reindex(delta.index).ffill()

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

        if blotter.empty:
            st.info("No trades above threshold.")
        else:
            # Save to session so other tabs (e.g., Blotter) can reuse it
            st.session_state["blotter_df"] = blotter
            st.session_state["blotter_meta"] = {
                "model": model,
                "thresh_bps": int(thresh_bps),
                "notional": int(notional),
            }

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
                key="signals_download_btn",
            )

            # Paper Trading (simulated)
            st.markdown("### 📊 Paper Trading")
            
            trading_panel(
                blotter,
                api_base="http://127.0.0.1:8000",
                use_api=True,
                key_prefix="signals_trade"
            )

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
                st.download_button("⬇️ Download return attribution", data=attr_ret.to_csv().encode(), file_name="attribution_return.csv")
            else:
                st.info("Return attribution file not found.")

        with colR:
            st.markdown("**Risk attribution**")
            if attr_risk is not None and not attr_risk.empty:
                if isinstance(attr_risk.columns, pd.MultiIndex) and attr_risk.columns.nlevels == 2:
                    models_risk = sorted(set(attr_risk.columns.get_level_values(0)))
                    default_idx = models_risk.index(model_ret) if "model_ret" in locals() and model_ret in models_risk else 0
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
                st.download_button("⬇️ Download risk attribution", data=attr_risk.to_csv().encode(), file_name="attribution_risk.csv")
            else:
                st.info("Risk attribution file not found.")


from app.components.comments import render_comments
from app.components.insights import render_insights
from app.components.reports import render_reports_panel

# inside your "Insights & Collaboration" tab:
with tab7:
    st.subheader("Insights & Collaboration")

    colA, colB = st.columns([2,1])
    with colA:
        thread_id = st.text_input(
            "Discussion thread",
            value="default",
            help="Use a preset/strategy name to keep threads organized.",
            key="collab_thread",
        )
    with colB:
        display_author = st.text_input("Your name", value="Researcher", key="collab_author")

    st.markdown("### AI Insights")
    render_insights(api_base=API_BASE, use_api=USE_API, key_prefix="insights_collab")

    st.markdown("### Comments")
    from app.components.comments import render_comments

    render_comments(
        thread_id=thread_id,                # whatever you collected from the UI
        author_default=display_author,      # your “Your name” value
        api_base=API_BASE,
        use_api=USE_API,
        key_prefix="comments_collab",       # unique key to avoid Streamlit key clashes
    )

    st.markdown("### Reports")
    render_reports_panel(api_base=API_BASE, use_api=USE_API, root=ROOT, key_prefix="reports_collab")
# In sidebar (or first location)
# (A) Wherever you render it the first time (e.g., main body)
# in the main body (existing call at ~line 829)
#render_reports_panel(api_base=API_BASE, use_api=USE_API, root=ROOT, key_prefix="reports_main")

# in your new tab (Insights & Collaboration)
#render_reports_panel(api_base=API_BASE, use_api=USE_API, root=ROOT, key_prefix="reports_collab")

# if you also render in the sidebar, use another:
# render_reports_panel(api_base=API_BASE, use_api=USE_API, root=ROOT, key_prefix="reports_sidebar")





with tab8:
    st.subheader("Universe")
    render_universe_editor(CFG_PATH)

    # optional: kick off the first stages if requested (API mode)
    if USE_API and st.session_state.get("trigger_run_fetch_features"):
        st.session_state.pop("trigger_run_fetch_features", None)
        with st.spinner("Queuing Fetch + Features…"):
            try:
                requests.post(f"{API_BASE}/run/queue", timeout=5)  # queues full run; or call /run/fetch+features if you prefer
                st.success("Queued.")
            except Exception as e:
                st.error(f"Queue failed: {e}")

with tab9:
    render_versions_panel(api_base=API_BASE, use_api=USE_API, cfg_path=CFG_PATH, key_prefix="versions_main")


with tab10:
    render_frontier(key_prefix="frontier_main")


with tab11:
    # pass API base to inner components so Reports works
    st.session_state["api_base"] = API_BASE
    render_dashboard_builder(key_prefix="builder_main")


with tab12:
    render_ml_panel(api_base=API_BASE, use_api=USE_API, root=ROOT, key_prefix="ml_main")

# ---------- Optional API: one-click run using managed YAML ----------
if USE_API:
    if st.sidebar.button("Run All for current strategy"):
        try:
            api_post("/run/all")
            st.sidebar.success("Run submitted. Refresh in a moment.")
        except Exception as e:
            st.sidebar.error(f"Run failed: {e}")


import streamlit as st







st.markdown(
    """
    <hr>
    <div style="text-align:center; font-size:14px; color:gray;">
    © 2025 Aditya Ravi — All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)
