# app/components/presets.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Any, Optional

import streamlit as st
import yaml
import requests


ROOT = Path(__file__).resolve().parents[2]
CFG_DIR = ROOT / "configs"
CFG_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CFG = CFG_DIR / "default.yaml"
PRESETS_DIR = CFG_DIR / "presets"
PRESETS_DIR.mkdir(parents=True, exist_ok=True)


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _save_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


def list_presets() -> list[str]:
    return sorted([p.stem for p in PRESETS_DIR.glob("*.yaml")])


def load_preset_to_default(name: str) -> Dict[str, Any]:
    """Copy preset YAML into default.yaml and return it."""
    src = PRESETS_DIR / f"{name}.yaml"
    data = _load_yaml(src)
    if not data:
        raise FileNotFoundError(f"Preset not found or empty: {src}")
    _save_yaml(DEFAULT_CFG, data)
    return data


def save_current_as_preset(name: str) -> None:
    """Save current default.yaml as a named preset."""
    data = _load_yaml(DEFAULT_CFG)
    if not data:
        data = {}
    _save_yaml(PRESETS_DIR / f"{name}.yaml", data)


def render_presets_panel(*, api_base: str, use_api: bool) -> Optional[str]:
    """
    Render sidebar UI for presets.
    Returns a job_id string if it triggered a background run; otherwise None.
    """
    st.markdown("### 🎛️ Presets")

    # row 1: select existing + load
    cols = st.columns([2, 1, 1])
    with cols[0]:
        preset_names = list_presets()
        sel = st.selectbox("Preset", preset_names, index=0 if preset_names else None, key="preset_sel")
    with cols[1]:
        auto_run = st.checkbox("Auto-run", value=True, help="Run the full pipeline after loading")
    with cols[2]:
        load_clicked = st.button("Load")

    job_id: Optional[str] = None

    if load_clicked and sel:
        try:
            data = load_preset_to_default(sel)
            st.success(f"Loaded preset '{sel}' into default.yaml")
            if auto_run and use_api:
                # queue background run (non-blocking)
                r = requests.post(f"{api_base}/run/queue", timeout=5)
                r.raise_for_status()
                job_id = r.json().get("job_id")
                st.info(f"Queued pipeline: job_id = {job_id}")
        except Exception as e:
            st.error(f"Load failed: {e}")

    # row 2: save current as new preset
    with st.expander("Save current config as a new preset"):
        new_name = st.text_input("Preset name", placeholder="e.g., research_2025Q3")
        if st.button("Save preset"):
            try:
                if not new_name.strip():
                    raise ValueError("Preset name cannot be empty")
                save_current_as_preset(new_name.strip())
                st.success(f"Saved preset '{new_name.strip()}'")
                st.rerun()
            except Exception as e:
                st.error(f"Save failed: {e}")

    return job_id