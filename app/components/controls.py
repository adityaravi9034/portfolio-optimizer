# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
# app/components/controls.py
from __future__ import annotations

from pathlib import Path
import os
import yaml
import requests
import streamlit as st

# Resolve project root: components/ -> app/ -> project/
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG_PATH = ROOT / "configs" / "default.yaml"

def _api_base() -> str:
    return os.getenv("OPTIM_API_URL", "").rstrip("/")

def _read_cfg(cfg_path: Path) -> dict:
    if not cfg_path.exists():
        return {"universe": {}, "constraints": {}}
    try:
        data = yaml.safe_load(cfg_path.read_text()) or {}
        data.setdefault("universe", {})
        data.setdefault("constraints", {})
        return data
    except Exception:
        return {"universe": {}, "constraints": {}}

def _write_cfg(cfg_path: Path, cfg: dict) -> None:
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

def render_strategy_controls(cfg_path: Path | None = None) -> None:
    """
    Render the strategy settings form.
    If cfg_path is provided, read/write that YAML; otherwise use DEFAULT_CFG_PATH.
    If OPTIM_API_URL is set, attempt to save via API; fallback to local write on error.
    """
    cfg_path = cfg_path or DEFAULT_CFG_PATH
    api_base = _api_base()
    use_api = bool(api_base)

    cfg = _read_cfg(cfg_path)
    uni = cfg.get("universe", {})
    cons = cfg.get("constraints", {})

    st.caption("Edit settings, then click **Save**.")
    with st.form("strategy_controls"):
        tickers_default = ",".join(
            uni.get("equities", [])
            or [
                "SPY","QQQ","EFA","EEM","IWM","VNQ","XLK","XLF","XLE","XLV","XLB","XLY","XLP",
                "IEF","TLT","LQD","HYG","SHY","TIP","BNDX","GLD","SLV","DBC","USO","BTC-USD","ETH-USD",
            ]
        )
        tickers_str = st.text_area(
            "Universe (comma-separated tickers)",
            value=tickers_default,
            height=100,
            help="Commas or newlines are ok.",
        )

        max_w = st.slider(
            "Max weight per asset",
            0.05,
            1.0,
            float(cons.get("max_weight", 0.30)),
            0.05,
        )
        long_only = st.checkbox(
            "Long only",
            bool(cons.get("long_only", True)),
        )
        cash_buffer = st.slider(
            "Cash buffer",
            0.0,
            0.20,
            float(cons.get("cash_buffer", 0.00)),
            0.01,
        )

        do_save = st.form_submit_button("Save settings")

    if do_save:
        # normalize tickers (support newlines/spaces)
        raw = tickers_str.replace("\n", ",")
        u = [s.strip() for s in raw.split(",") if s.strip()]

        cfg.setdefault("universe", {})["equities"] = u
        cfg.setdefault("constraints", {})
        cfg["constraints"]["max_weight"] = float(max_w)
        cfg["constraints"]["long_only"] = bool(long_only)
        cfg["constraints"]["cash_buffer"] = float(cash_buffer)

        if use_api:
            try:
                r = requests.post(
                    f"{api_base}/config",
                    json={
                        "universe": u,
                        "max_weight": float(max_w),
                        "long_only": bool(long_only),
                        "cash_buffer": float(cash_buffer),
                    },
                    timeout=10,
                )
                r.raise_for_status()
                st.success("Config saved via API.")
            except Exception as e:
                # fallback to local write
                _write_cfg(cfg_path, cfg)
                st.warning(f"API save failed, wrote YAML locally. ({e})")
        else:
            _write_cfg(cfg_path, cfg)
            st.success(f"Config saved to {cfg_path}")