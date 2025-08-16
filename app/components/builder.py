# app/components/builder.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import streamlit as st

# Re-use your existing widgets
from app.components.live_tiles import render_live_kpis
from app.components.charts import plot_equity_curves, plot_weights_area
from app.components.insights import render_insights
from app.components.reports import render_reports_panel
from app.components.controls import render_strategy_controls

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)
DB_FILE = DATA / "dashboards.json"

# ---------------- Registry of available widgets ----------------
def _w_live_kpis(key_prefix: str): render_live_kpis()
def _w_equity(key_prefix: str):     plot_equity_curves()
def _w_weights(key_prefix: str):    plot_weights_area()
def _w_insights(key_prefix: str):   render_insights(key_prefix=f"{key_prefix}_ins")
def _w_reports(key_prefix: str):    render_reports_panel(
    api_base=st.session_state.get("API_BASE",""),
    use_api=bool(st.session_state.get("API_BASE")),
    root=ROOT,
    key_prefix=f"{key_prefix}_rep",
)
def _w_controls(key_prefix: str):   render_strategy_controls()

WIDGETS = {
    "Live KPIs": _w_live_kpis,
    "Equity Curves": _w_equity,
    "Weights Area": _w_weights,
    "AI Insights": _w_insights,
    "Reports": _w_reports,
    "Strategy Controls": _w_controls,
}

# ---------------- Layout model ----------------
@dataclass
class DashboardLayout:
    name: str
    layout: int                 # 1,2,3 columns
    widgets: list[str]          # ordered

# ---------------- Persistence ----------------
def _load_all() -> dict[str, DashboardLayout]:
    if not DB_FILE.exists():
        return {}
    try:
        raw = json.loads(DB_FILE.read_text())
        out = {}
        for k,v in raw.items():
            out[k] = DashboardLayout(name=v["name"], layout=int(v["layout"]), widgets=list(v["widgets"]))
        return out
    except Exception:
        return {}

def _save_all(objs: dict[str, DashboardLayout]) -> None:
    DB_FILE.write_text(json.dumps({k: asdict(v) for k,v in objs.items()}, indent=2))

# ---------------- Renderer ----------------
def render_dashboard_builder(key_prefix: str = "builder"):
    st.subheader("🧩 Custom Dashboard Builder (beta)")

    # make API_BASE available to internal widgets
    # (set once from caller)
    if "API_BASE" not in st.session_state:
        st.session_state["API_BASE"] = st.session_state.get("api_base", "")

    saved = _load_all()
    existing_names = sorted(saved.keys())

    # Load / New row
    c1, c2 = st.columns([2,1])
    with c1:
        sel = st.selectbox("Load existing layout", ["— new —"] + existing_names, index=0, key=f"{key_prefix}_load")
    with c2:
        name = st.text_input("Layout name", value=("My Dashboard" if sel == "— new —" else sel), key=f"{key_prefix}_name")

    # Base layout choice
    cols_opt = st.radio("Columns", [1,2,3], horizontal=True, index=1, key=f"{key_prefix}_cols")

    # Widget chooser (ordered via multiselect order)
    default_widgets = list(WIDGETS.keys())[:3]
    prefill = saved[sel].widgets if sel in saved else default_widgets
    picked = st.multiselect(
        "Widgets (drag to order)",
        list(WIDGETS.keys()),
        default=prefill,
        key=f"{key_prefix}_widgets",
    )

    st.markdown("---")

    # Actions
    a1, a2, a3 = st.columns(3)
    with a1:
        if st.button("💾 Save", type="primary", key=f"{key_prefix}_save"):
            if not name.strip():
                st.warning("Enter a layout name.")
            elif not picked:
                st.warning("Pick at least one widget.")
            else:
                saved[name] = DashboardLayout(name=name.strip(), layout=int(cols_opt), widgets=picked)
                _save_all(saved)
                st.success(f"Saved layout '{name}'.")
    with a2:
        if sel in saved and st.button("🗑️ Delete", key=f"{key_prefix}_delete"):
            saved.pop(sel, None)
            _save_all(saved)
            st.warning(f"Deleted '{sel}'.")
            st.rerun()
    with a3:
        st.download_button(
            "⬇️ Export JSON",
            data=json.dumps({k: asdict(v) for k,v in saved.items()}, indent=2).encode(),
            file_name="dashboards.json",
            key=f"{key_prefix}_export"
        )

    st.markdown("---")
    st.markdown("### Preview")

    # Render preview
    if not picked:
        st.info("Pick some widgets to preview.")
        return

    cols = st.columns(cols_opt)
    # round-robin placement
    for i, w in enumerate(picked):
        target = cols[i % cols_opt]
        with target:
            st.markdown(f"##### {w}")
            try:
                WIDGETS[w](key_prefix=f"{key_prefix}_{i}")
            except Exception as e:
                st.error(f"Widget '{w}' failed: {e}")

    st.caption("Tip: Save the layout and add a quick-switch in the Builder tab to load later.")