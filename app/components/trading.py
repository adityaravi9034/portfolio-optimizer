# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
from __future__ import annotations
from typing import List, Dict, Tuple
import pandas as pd
import requests
import streamlit as st


def _orders_from_blotter(blotter: pd.DataFrame) -> List[Dict]:
    """Map blotter dataframe to API order schema. Skips ~0 share rows."""
    if blotter is None or blotter.empty:
        return []

    cols = {c.lower(): c for c in blotter.columns}
    def col(name): return cols.get(name, name)

    out: List[Dict] = []
    for _, r in blotter.iterrows():
        try:
            sh = float(r[col("approx_shares")])
        except Exception:
            sh = 0.0
        if abs(sh) < 1e-9:
            continue
        out.append({
            "asset":         r[col("asset")],
            "approx_shares": sh,
            "last_price":    None if "last_price" not in cols else float(r[col("last_price")]),
            "dollars":       None if "dollars"    not in cols else float(r[col("dollars")]),
        })
    return out


def _disabled_reason(use_api: bool, api_base: str, orders: List[Dict]) -> Tuple[bool, str]:
    if not use_api:
        return True, "use_api=False"
    if not api_base:
        return True, "api_base is empty"
    if not orders:
        return True, "no non-zero orders"
    return False, ""


def trading_panel(
    blotter: pd.DataFrame,
    api_base: str,
    use_api: bool = True,
    key_prefix: str = "trade",
):
    """
    Paper trading panel with explicit diagnostics and force-rerun after submit.
    Requires api_base like 'http://127.0.0.1:8000' and use_api=True.
    """
    st.header("📊 Paper Trading")

    # --- Inputs & diagnostics ---
    orders = _orders_from_blotter(blotter)
    st.caption(f"API base: `{api_base or '—'}` • use_api={use_api} • Orders ready: **{len(orders)}**")
    st.write("Preview payload:")
    st.json({"orders": orders})

    disabled, reason = _disabled_reason(use_api, api_base, orders)
    if disabled:
        st.info(f"Buttons disabled — reason: **{reason}**. "
                "Tip: set Threshold=0, Round lots OFF, Lot size=1, and/or increase Notional to get non-zero orders.")

    api = api_base.rstrip("/")

    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        if st.button("Preview impact", key=f"{key_prefix}_preview", use_container_width=True, disabled=disabled):
            try:
                r = requests.post(f"{api}/broker/paper/preview", json={"orders": orders}, timeout=10)
                r.raise_for_status()
                data = r.json()
                gross = float(data.get("est_gross_dollars", 0))
                n = int(data.get("n_orders", len(orders)))
                st.success(f"Estimated gross notional: ${gross:,.0f} across {n} orders")
            except Exception as e:
                st.error(f"Preview failed: {e}")

    with c2:
        if st.button("Execute orders", key=f"{key_prefix}_execute", type="primary",
                     use_container_width=True, disabled=disabled):
            try:
                r = requests.post(f"{api}/broker/execute", json={"orders": orders}, timeout=15)
                r.raise_for_status()
                data = r.json()
                n = int(data.get("n_orders", len(orders)))
                st.success(f"Submitted {n} orders ✅")
                st.rerun()  # pull fresh orders/positions below
            except Exception as e:
                st.error(f"Execute failed: {e}")

    with c3:
        # Always enabled: lets you verify wiring even if orders==0.
        if st.button("Debug: force execute 1 AAPL", key=f"{key_prefix}_force", use_container_width=True):
            try:
                test = {"orders": [{"asset": "AAPL", "approx_shares": 1, "last_price": 190.0, "dollars": 190.0}]}
                r = requests.post(f"{api}/broker/execute", json=test, timeout=10)
                r.raise_for_status()
                st.success("Force execute sent ✅")
                st.rerun()
            except Exception as e:
                st.error(f"Force execute failed: {e}")

    with st.expander("Live orders & positions", expanded=True):
        colL, colR = st.columns(2)
        with colL:
            st.write("**Positions**")
            try:
                rp = requests.get(f"{api}/broker/positions", timeout=5)
                if rp.ok:
                    # API returns {"ok": true, "items": [{"asset":"AAPL","shares":5.0}]} or {"positions": {...}}
                    data = rp.json()
                    if isinstance(data, dict):
                        st.json(data.get("items") or data.get("positions") or data)
                    else:
                        st.json({"positions": data})
                else:
                    st.error(f"Positions HTTP {rp.status_code}")
            except Exception as e:
                st.error(f"Positions error: {e}")
        with colR:
            st.write("**Recent orders**")
            try:
                ro = requests.get(f"{api}/broker/orders", timeout=5)
                if ro.ok:
                    data = ro.json()
                    items = data.get("items") if isinstance(data, dict) else data
                    st.json({"items": items})
                else:
                    st.error(f"Orders HTTP {ro.status_code}")
            except Exception as e:
                st.error(f"Orders error: {e}")

        st.button("Refresh", key=f"{key_prefix}_refresh", on_click=lambda: st.rerun())