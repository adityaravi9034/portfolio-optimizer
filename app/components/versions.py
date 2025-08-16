from __future__ import annotations
import yaml, requests, streamlit as st
from pathlib import Path

def _safe_yaml_load(text: str):
    try:
        return yaml.safe_load(text) or {}
    except Exception:
        return {}

def _diff_dict(a: dict, b: dict, prefix=""):
    diffs = []
    keys = sorted(set(a.keys()) | set(b.keys()))
    for k in keys:
        pa = a.get(k, None)
        pb = b.get(k, None)
        pfx = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(pa, dict) and isinstance(pb, dict):
            diffs += _diff_dict(pa, pb, pfx)
        elif pa != pb:
            diffs.append((pfx, pa, pb))
    return diffs

def render_versions_panel(*, api_base: str, use_api: bool, cfg_path: Path, key_prefix: str = "versions"):
    st.markdown("### 🗂️ Strategy Versions")
    if not use_api:
        st.info("API mode off. Set OPTIM_API_URL to enable versioning.")
        return

    # Choose logical strategy id (use the YAML 'name' or a text box)
    cfg = _safe_yaml_load(cfg_path.read_text()) if cfg_path.exists() else {}
    default_name = (cfg.get("meta", {}) or {}).get("name") or "default"
    colA, colB = st.columns([2,1])
    with colA:
        strategy = st.text_input("Strategy name", value=default_name, key=f"{key_prefix}_name")
    with colB:
        author = st.text_input("Author", value="Researcher", key=f"{key_prefix}_author")

    # Save current YAML as new version
    note = st.text_input("Note (optional)", placeholder="What changed?", key=f"{key_prefix}_note")
    if st.button("Save snapshot", type="primary", key=f"{key_prefix}_save"):
        try:
            payload = {
                "strategy": strategy.strip(),
                "author": author.strip(),
                "note": note.strip() or None,
                "yaml_text": cfg_path.read_text(),
            }
            r = requests.post(f"{api_base}/versions", json=payload, timeout=15)
            r.raise_for_status()
            st.success(f"Saved version #{r.json().get('id')}")
            st.rerun()
        except Exception as e:
            st.error(f"Save failed: {e}")

    # List versions
    items = []
    try:
        r = requests.get(f"{api_base}/versions/{strategy}", timeout=10)
        r.raise_for_status()
        items = r.json().get("items", [])
    except Exception as e:
        st.error(f"Load versions failed: {e}")

    if not items:
        st.info("No versions yet.")
        return

    st.divider()
    st.write("**Recent versions**")
    ids = [str(it["id"]) for it in items]
    col1, col2, col3 = st.columns([1,1,2])

    with col1:
        vA = st.selectbox("Version A", ids, key=f"{key_prefix}_vA")
    with col2:
        vB = st.selectbox("Version B", ids, index=min(1, len(ids)-1), key=f"{key_prefix}_vB")
    with col3:
        st.caption("Pick two to diff; load any to YAML.")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Diff A vs B", key=f"{key_prefix}_diff"):
            try:
                ya = requests.get(f"{api_base}/versions/get/{vA}", timeout=10).json().get("yaml","")
                yb = requests.get(f"{api_base}/versions/get/{vB}", timeout=10).json().get("yaml","")
                a = _safe_yaml_load(ya); b = _safe_yaml_load(yb)
                diffs = _diff_dict(a, b)
                if not diffs:
                    st.success("No differences.")
                else:
                    for path, left, right in diffs[:200]:
                        st.write(f"**{path}**")
                        st.code(f"A: {left}\nB: {right}")
            except Exception as e:
                st.error(f"Diff failed: {e}")
    with c2:
        if st.button("Load A into YAML", key=f"{key_prefix}_loadA"):
            try:
                ya = requests.get(f"{api_base}/versions/get/{vA}", timeout=10).json().get("yaml","")
                Path(cfg_path).write_text(ya)
                st.success(f"Loaded version {vA} into YAML. Re-run stages to apply.")
            except Exception as e:
                st.error(f"Load failed: {e}")
    with c3:
        if st.button("Load B into YAML", key=f"{key_prefix}_loadB"):
            try:
                yb = requests.get(f"{api_base}/versions/get/{vB}", timeout=10).json().get("yaml","")
                Path(cfg_path).write_text(yb)
                st.success(f"Loaded version {vB} into YAML. Re-run stages to apply.")
            except Exception as e:
                st.error(f"Load failed: {e}")

    st.markdown("#### Table")
    st.dataframe(items, use_container_width=True)