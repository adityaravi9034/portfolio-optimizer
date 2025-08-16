# app/components/charts.py
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# charts.py is inside app/components/, so go up TWO levels to project root
ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"

# Toggle at runtime via set_debug(True/False)
DEBUG_CHARTS = False


# ---------- Debug helpers ----------
def set_debug(flag: bool) -> None:
    global DEBUG_CHARTS
    DEBUG_CHARTS = bool(flag)


def _debug(msg: str) -> None:
    if DEBUG_CHARTS:
        st.caption(f"🛠 {msg}")


# ---------- CSV readers ----------
def _read_csv_forgiving(path: Path, parse_dates: bool = False) -> Optional[pd.DataFrame]:
    """Read CSV with fallback behavior and robust datetime coercion."""
    if not path.exists() or path.stat().st_size == 0:
        _debug(f"{path} missing/empty")
        return None

    # First attempt
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=parse_dates)
        if parse_dates and not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]
        _debug(f"read {path.name} → {df.shape}")
        return df
    except Exception as e:
        _debug(f"read_csv failed ({path.name}): {e}")

    # Fallback attempt
    try:
        df = pd.read_csv(path, index_col=0)
        if parse_dates:
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]
        _debug(f"fallback read {path.name} → {df.shape}")
        return df
    except Exception as e:
        _debug(f"fallback failed ({path.name}): {e}")
        return None


def _read_weights_timeseries(path: Path) -> Optional[pd.DataFrame]:
    """Try MultiIndex header first; fall back to single header."""
    if not path.exists() or path.stat().st_size == 0:
        _debug(f"{path} missing/empty")
        return None

    # MultiIndex attempt
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2:
            if not pd.api.types.is_datetime64_any_dtype(df.index):
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df[~df.index.isna()]
            _debug(f"read multi {path.name} → {df.shape}")
            return df
    except Exception as e:
        _debug(f"multi read failed: {e}")

    # Single header fallback
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()]
        _debug(f"read single {path.name} → {df.shape}")
        return df
    except Exception as e:
        _debug(f"single read failed: {e}")
        return None


# ---------- Equity helpers & charts ----------
def equity_models() -> List[str]:
    eq = _read_csv_forgiving(PROC / "equity_curves.csv", parse_dates=True)
    return [] if (eq is None or eq.empty) else list(eq.columns)


def plot_equity_curves(models_sel: Optional[List[str]] = None, key_prefix: str = "charts_eq") -> None:
    eq = _read_csv_forgiving(PROC / "equity_curves.csv", parse_dates=True)
    if eq is None or eq.empty:
        st.info("No equity curves yet.")
        return

    models = list(eq.columns)
    sel = models_sel or st.multiselect(
        "Models", models, default=models, key=f"{key_prefix}_models"
    )
    if not sel:
        st.warning("Select at least one model.")
        return

    # Equity curves
    fig = go.Figure()
    for m in sel:
        fig.add_trace(
            go.Scatter(
                x=eq.index, y=eq[m],
                mode="lines", name=m,
                hovertemplate="%{x}<br>%{y:.3f}x",
            )
        )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Equity (x)",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_plot")

    # Drawdowns (for selected models)
    dd = eq[sel] / eq[sel].cummax() - 1.0
    fig2 = go.Figure()
    for m in sel:
        fig2.add_trace(
            go.Scatter(
                x=dd.index, y=dd[m],
                mode="lines", name=m,
                hovertemplate="%{x}<br>%{y:.2%}",
            )
        )
    fig2.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Drawdown",
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
    )
    st.plotly_chart(fig2, use_container_width=True, key=f"{key_prefix}_dd_plot")


def plot_equity_compare(model_a: str, model_b: str, key_prefix: str = "charts_eq_cmp") -> None:
    eq = _read_csv_forgiving(PROC / "equity_curves.csv", parse_dates=True)
    if eq is None or eq.empty or model_a not in eq or model_b not in eq:
        st.info("Select two valid models to compare.")
        return

    c1, c2 = st.columns(2)
    for col, m in zip((c1, c2), (model_a, model_b)):
        with col:
            st.markdown(f"**{m} — Equity & Drawdown**")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eq.index, y=eq[m], mode="lines", name=m))
            fig.update_layout(
                height=280, margin=dict(l=10, r=10, t=30, b=10),
                yaxis_title="Equity (x)"
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_{m}_eq")

            dd = eq[m] / eq[m].cummax() - 1.0
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=dd.index, y=dd, mode="lines", name=f"{m} DD"))
            fig2.update_layout(
                height=220, margin=dict(l=10, r=10, t=30, b=10),
                yaxis_title="Drawdown"
            )
            st.plotly_chart(fig2, use_container_width=True, key=f"{key_prefix}_{m}_dd")


# ---------- Weights helpers & charts ----------
def weights_models() -> List[str]:
    wt = _read_weights_timeseries(PROC / "weights_timeseries.csv")
    if wt is None or wt.empty:
        return []
    if isinstance(wt.columns, pd.MultiIndex):
        return wt.columns.get_level_values(0).unique().tolist()
    # single header fallback — expose a pseudo-model
    return ["Model"]


def plot_weights_area(key_prefix: str = "weights_section") -> None:
    """Render stacked area of weights with unique widget keys via key_prefix."""
    wt_path = PROC / "weights_timeseries.csv"
    wt = _read_weights_timeseries(wt_path)
    if wt is None or wt.empty:
        st.info("No weights history yet.")
        return

    # normalize to single-model DataFrame
    if isinstance(wt.columns, pd.MultiIndex):
        models = wt.columns.get_level_values(0).unique().tolist()
        model = st.selectbox(
            "Model (weights)", models, index=0, key=f"{key_prefix}_model"
        )
        df = wt[model].copy()
    else:
        df = wt.copy()

    assets_all = df.columns.tolist()
    sel_assets = st.multiselect(
        "Assets", assets_all, default=assets_all, key=f"{key_prefix}_assets"
    )
    if not sel_assets:
        st.warning("Select at least one asset.")
        return

    df = df[sel_assets].fillna(0.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df.sum(axis=1),
            mode="lines", name="Total",
            line=dict(width=1),
            hovertemplate="%{x}<br>Total: %{y:.2%}",
        )
    )
    for c in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[c],
                stackgroup="one", name=c,
                hovertemplate="%{x}<br>%{y:.2%}",
            )
        )
    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_plot")


def plot_weights_compare(model_a: str, model_b: str, key_prefix: str = "weights_cmp") -> None:
    wt = _read_weights_timeseries(PROC / "weights_timeseries.csv")
    if wt is None or wt.empty or not isinstance(wt.columns, pd.MultiIndex):
        st.info("Compare requires weights_timeseries with (model, asset) columns.")
        return
    if model_a not in wt.columns.get_level_values(0) or model_b not in wt.columns.get_level_values(0):
        st.info("Select two valid models.")
        return

    c1, c2 = st.columns(2)
    for col, m in zip((c1, c2), (model_a, model_b)):
        with col:
            st.markdown(f"**{m} — Weights (stacked)**")
            df = wt[m].fillna(0.0)
            assets_all = df.columns.tolist()
            sel_assets = st.multiselect(
                "Assets", assets_all, default=assets_all, key=f"{key_prefix}_assets_{m}"
            )
            if not sel_assets:
                st.warning("Select at least one asset.")
                continue

            dfx = df[sel_assets]
            fig = go.Figure()
            for c in dfx.columns:
                fig.add_trace(
                    go.Scatter(
                        x=dfx.index, y=dfx[c],
                        stackgroup="one", name=c,
                        hovertemplate="%{x}<br>%{y:.2%}",
                    )
                )
            fig.update_layout(
                height=420, margin=dict(l=10, r=10, t=30, b=10),
                yaxis_tickformat=".0%",
            )
            st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_plot_{m}")