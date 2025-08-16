# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
# app/components/frontier.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
TRADING_DAYS = 252


# ---------- lightweight loaders (cached) ----------
@st.cache_data
def _load_equity(eq_path: str, *, mtime: int, size: int) -> pd.DataFrame | None:
    """
    Load equity_curves.csv. Cache key includes file mtime/size so it refreshes when file changes.
    """
    p = Path(eq_path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    # drop rows with NA in any series (keeps rolling math simple)
    df = df[df.notna().all(axis=1)]
    return df


@st.cache_data
def _load_weights_ts(w_path: str, *, mtime: int, size: int) -> pd.DataFrame | None:
    """
    Load weights_timeseries.csv. Handles MultiIndex headers.
    Cache key includes mtime/size so it refreshes when file changes.
    """
    p = Path(w_path)
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(p, header=[0, 1], index_col=0, parse_dates=True)
        if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2:
            return df
    except Exception:
        pass
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    return df


# ---------- helpers ----------
def _metrics_from_returns(r: pd.Series) -> dict:
    if r is None or r.empty:
        return {"ret": 0.0, "vol": 0.0, "sharpe": 0.0, "mdd": 0.0}
    mu = r.mean() * TRADING_DAYS
    vol = r.std(ddof=0) * np.sqrt(TRADING_DAYS)
    sharpe = (mu / vol) if vol > 0 else 0.0
    eq = (1 + r).cumprod()
    mdd = (eq / eq.cummax() - 1).min()
    return {"ret": float(mu), "vol": float(vol), "sharpe": float(sharpe), "mdd": float(mdd)}


def _resample_weights(weights_ts: pd.DataFrame, model: str | None) -> pd.DataFrame:
    """
    Build a matrix of candidate portfolios by sampling historical weight rows.
    If MultiIndex, pick one model slice; otherwise use all rows.
    """
    if isinstance(weights_ts.columns, pd.MultiIndex):
        models = sorted(set(weights_ts.columns.get_level_values(0)))
        if not model:
            model = models[0]
        W = weights_ts[model].copy()
    else:
        W = weights_ts.copy()

    W = W.fillna(0.0)
    W[W < 0] = 0  # assume long-only frontier
    row_sums = W.sum(axis=1).replace(0, np.nan)
    W = W.div(row_sums, axis=0).fillna(0.0)
    return W


@st.cache_data
def _compute_frontier_points(
    *,
    eq_path: str,
    eq_mtime: int,
    eq_size: int,
    choices: tuple[str, ...],
    window: int,
    stride: int,
) -> pd.DataFrame:
    """
    Compute rolling-window metrics that drive the 3D scatter.

    Cached and keyed by:
      - equity file (path + mtime + size)
      - the selected curve names (choices)
      - the slider values (window, stride)
    """
    # load fresh inside the cache to tie to eq_mtime/eq_size
    eq = _load_equity(eq_path, mtime=eq_mtime, size=eq_size)
    if eq is None or eq.empty:
        return pd.DataFrame()

    pts = []
    for col in choices:
        if col not in eq.columns:
            continue
        r = eq[col].pct_change().dropna()
        if len(r) <= window:
            # still compute one window if it barely fits
            rr = r.iloc[-window:] if len(r) > 0 else r
            m = _metrics_from_returns(rr)
            pts.append({"name": col, **m})
            continue

        # rolling windows with given stride
        for start in range(0, len(r) - window + 1, max(1, stride)):
            rr = r.iloc[start : start + window]
            m = _metrics_from_returns(rr)
            pts.append({"name": col, **m})

    return pd.DataFrame(pts)


# ---------- main component ----------
def render_frontier(key_prefix: str = "frontier"):
    st.subheader("Efficient Frontier (3D)")

    # paths + cache-bust keys
    p_eq = PROC / "equity_curves.csv"
    p_wt = PROC / "weights_timeseries.csv"
    eq_mtime, eq_size = (int(p_eq.stat().st_mtime), int(p_eq.stat().st_size)) if p_eq.exists() else (0, 0)
    wt_mtime, wt_size = (int(p_wt.stat().st_mtime), int(p_wt.stat().st_size)) if p_wt.exists() else (0, 0)

    eq = _load_equity(str(p_eq), mtime=eq_mtime, size=eq_size)
    wt_ts = _load_weights_ts(str(p_wt), mtime=wt_mtime, size=wt_size)

    if eq is None or wt_ts is None or eq.empty or wt_ts.empty:
        st.info("Need equity_curves.csv and weights_timeseries.csv. Run Walkforward/Backtest first.")
        return

    # choose a base model
    if isinstance(wt_ts.columns, pd.MultiIndex):
        models = sorted(set(wt_ts.columns.get_level_values(0)))
    else:
        models = list(eq.columns)
    model = st.selectbox("Model context", models, index=0, key=f"{key_prefix}_model")

    # (kept for future use if you want to synthesize portfolios from weights)
    _ = _resample_weights(wt_ts, model)

    # curve selection
    choices = st.multiselect(
        "Curves to include",
        list(eq.columns),
        default=list(eq.columns),
        key=f"{key_prefix}_curves",
    )
    if not choices:
        st.warning("Select at least one curve.")
        return

    # sliders – these now drive the cache key of the computation
    window = st.slider("Lookback (days) for metrics", 60, 756, 252, step=21, key=f"{key_prefix}_win")
    stride = st.slider("Stride (days)", 5, 63, 21, step=1, key=f"{key_prefix}_stride")

    # compute
    df = _compute_frontier_points(
        eq_path=str(p_eq),
        eq_mtime=eq_mtime,
        eq_size=eq_size,
        choices=tuple(choices),
        window=int(window),
        stride=int(stride),
    )

    if df.empty:
        st.info("Not enough data for the chosen lookback.")
        return

    # 3D scatter: x=Vol, y=Return, z=|MaxDD| (positive number), color=Sharpe
    fig = go.Figure()
    for name, grp in df.groupby("name"):
        fig.add_trace(
            go.Scatter3d(
                x=grp["vol"],
                y=grp["ret"],
                z=grp["mdd"].abs(),
                mode="markers",
                name=name,
                marker=dict(size=4, color=grp["sharpe"], colorscale="Viridis", showscale=False),
                hovertemplate=(
                    "Curve: %{meta}<br>"
                    "Vol: %{x:.2%}<br>"
                    "Ret: %{y:.2%}<br>"
                    "|MaxDD|: %{z:.2%}<br>"
                    "Sharpe: %{marker.color:.2f}<extra></extra>"
                ),
                meta=name,
            )
        )

    fig.update_layout(
        height=520,
        margin=dict(l=5, r=5, t=30, b=5),
        scene=dict(
            xaxis_title="Volatility (σ, annualized)",
            yaxis_title="Return (μ, annualized)",
            zaxis_title="|Max Drawdown|",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Each point = a rolling window snapshot. Color = Sharpe. "
        "Z = absolute drawdown (lower is better)."
    )