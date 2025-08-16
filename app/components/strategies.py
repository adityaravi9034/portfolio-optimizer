# app/components/strategies.py
from __future__ import annotations
from pathlib import Path
import os, yaml, requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
CFG_PATH = ROOT / "configs" / "default.yaml"

API_BASE = os.getenv("OPTIM_API_URL", "").rstrip("/")
USE_API = bool(API_BASE)

# --- local fallback if API is off: store .yaml files in data/strategies/ ---
STRAT_DIR = ROOT / "data" / "strategies"
STRAT_DIR.mkdir(parents=True, exist_ok=True)

def _read_cfg():
    if not CFG_PATH.exists():
        return {"universe": {}, "constraints": {}}
    return yaml.safe_load(CFG_PATH.read_text()) or {"universe": {}, "constraints": {}}

def _write_cfg(cfg: dict):
    CFG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False))

def _api_list():
    r = requests.get(f"{API_BASE}/strategies", timeout=10)
    r.raise_for_status()
    return r.json().get("items", [])

def _api_get(name: str):
    r = requests.get(f"{API_BASE}/strategies/{name}", timeout=10)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json().get("item")

def _api_upsert(name: str, universe: list[str], max_weight: float, long_only: bool, cash_buffer: float):
    r = requests.post(f"{API_BASE}/strategies", json={
        "name": name,
        "universe": universe,
        "max_weight": max_weight,
        "long_only": long_only,
        "cash_buffer": cash_buffer,
    }, timeout=10)
    r.raise_for_status()
    return True

def _api_delete(name: str):
    r = requests.delete(f"{API_BASE}/strategies/{name}", timeout=10)
    r.raise_for_status()
    return True

def _local_list():
    out = []
    for p in sorted(STRAT_DIR.glob("*.yaml")):
        try:
            d = yaml.safe_load(p.read_text()) or {}
            out.append({
                "name": p.stem,
                "universe": d.get("universe", []),
                "max_weight": d.get("max_weight", 0.3),
                "long_only": d.get("long_only", True),
                "cash_buffer": d.get("cash_buffer", 0.0),
                "updated_at": p.stat().st_mtime,
            })
        except Exception:
            pass
    return out

def _local_get(name: str):
    p = STRAT_DIR / f"{name}.yaml"
    if not p.exists():
        return None
    return yaml.safe_load(p.read_text()) or {}

def _local_upsert(name: str, universe: list[str], max_weight: float, long_only: bool, cash_buffer: float):
    p = STRAT_DIR / f"{name}.yaml"
    p.write_text(yaml.safe_dump({
        "name": name,
        "universe": universe,
        "max_weight": float(max_weight),
        "long_only": bool(long_only),
        "cash_buffer": float(cash_buffer),
    }, sort_keys=False))
    return True

def _local_delete(name: str):
    p = STRAT_DIR / f"{name}.yaml"
    if p.exists():
        p.unlink()

def render_strategy_manager():
    st.subheader("💾 Strategy presets")

    # Current cfg preview
    cfg = _read_cfg()
    uni = cfg.get("universe", {}).get("equities", [])
    con = cfg.get("constraints", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Assets in universe", len(uni) if isinstance(uni, list) else 0)
    c2.metric("Max weight", f"{float(con.get('max_weight', 0.30)):.0%}")
    c3.metric("Long only", "Yes" if bool(con.get("long_only", True)) else "No")
    c4.metric("Cash buffer", f"{float(con.get('cash_buffer', 0.0)):.0%}")

    st.markdown("Save the **current YAML** as a reusable preset, or load an existing one.")

    # Save preset
    with st.expander("Save current as new preset", expanded=False):
        name = st.text_input("Preset name", placeholder="e.g. core_60_40")
        if st.button("Save preset", disabled=not name):
            if isinstance(uni, list):
                if USE_API:
                    try:
                        _api_upsert(name, uni, float(con.get("max_weight", 0.30)),
                                    bool(con.get("long_only", True)),
                                    float(con.get("cash_buffer", 0.0)))
                        st.success(f"Saved preset '{name}' via API")
                    except Exception as e:
                        st.error(f"API save failed: {e}")
                else:
                    _local_upsert(name, uni, float(con.get("max_weight", 0.30)),
                                  bool(con.get("long_only", True)),
                                  float(con.get("cash_buffer", 0.0)))
                    st.success(f"Saved preset '{name}' locally")
            else:
                st.error("Universe format invalid in YAML (expected list)")

    # List + load/delete
    st.markdown("### Presets")
    try:
        items = _api_list() if USE_API else _local_list()
    except Exception as e:
        items = []
        st.error(f"Could not load presets list: {e}")

    if not items:
        st.info("No presets yet. Save one above.")
        return

    # Simple table + actions
    for it in items:
        with st.container(border=True):
            left, right = st.columns([3,1])
            with left:
                st.write(f"**{it['name']}**")
                st.caption(f"{len(it.get('universe', []))} assets • max_w {it.get('max_weight', 0.3):.0%} • "
                           f"long_only {'Yes' if it.get('long_only', True) else 'No'} • "
                           f"cash {it.get('cash_buffer', 0.0):.0%}")
            with right:
                c1, c2 = st.columns(2)
                if c1.button("Load", key=f"load_{it['name']}"):
                    # Load preset and write to configs/default.yaml
                    if USE_API:
                        try:
                            item = _api_get(it["name"])
                            if not item:
                                st.error("Preset not found.")
                            else:
                                cfg.setdefault("universe", {})["equities"] = item["universe"]
                                cfg.setdefault("constraints", {})
                                cfg["constraints"]["max_weight"] = float(item["max_weight"])
                                cfg["constraints"]["long_only"] = bool(item["long_only"])
                                cfg["constraints"]["cash_buffer"] = float(item["cash_buffer"])
                                _write_cfg(cfg)
                                st.success(f"Loaded preset '{it['name']}' into YAML")
                        except Exception as e:
                            st.error(f"Load failed: {e}")
                    else:
                        item = _local_get(it["name"])
                        if not item:
                            st.error("Preset not found.")
                        else:
                            cfg.setdefault("universe", {})["equities"] = item["universe"]
                            cfg.setdefault("constraints", {})
                            cfg["constraints"]["max_weight"] = float(item["max_weight"])
                            cfg["constraints"]["long_only"] = bool(item["long_only"])
                            cfg["constraints"]["cash_buffer"] = float(item["cash_buffer"])
                            _write_cfg(cfg)
                            st.success(f"Loaded preset '{it['name']}' into YAML")

                if c2.button("Delete", key=f"del_{it['name']}"):
                    try:
                        if USE_API:
                            _api_delete(it["name"])
                        else:
                            _local_delete(it["name"])
                        st.success(f"Deleted preset '{it['name']}'")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Delete failed: {e}")