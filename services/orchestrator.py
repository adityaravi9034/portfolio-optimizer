# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
# services/orchestrator.py
import subprocess, sys, json, time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PY = str((ROOT / ".venv/bin/python").resolve()) if (ROOT / ".venv/bin/python").exists() else sys.executable
CFG = str((ROOT / "configs/default.yaml").resolve())

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATUS_PATH = DATA_DIR / "run_status.json"

STAGES = [
    ("fetch_market",     ["-m", "pipeline.fetch_market",    "--config", CFG]),
    ("build_features",   ["-m", "pipeline.build_features",  "--config", CFG]),
    ("optimize",         ["-m", "pipeline.optimize",        "--config", CFG]),
    ("walkforward",      ["-m", "pipeline.walkforward",     "--config", CFG]),
    ("stress",           ["-m", "pipeline.stress",          "--config", CFG]),
    ("backtest",         ["-m", "pipeline.backtest",        "--config", CFG]),
    ("attribution",      ["-m", "pipeline.attribution",     "--config", CFG]),
]

def _now_iso():
    return datetime.now(timezone.utc).isoformat()

def _read_status():
    if not STATUS_PATH.exists():
        return {"state":"idle","stage":None,"progress":0,"started_at":None,"updated_at":_now_iso()}
    try:
        return json.loads(STATUS_PATH.read_text())
    except Exception:
        return {"state":"idle","stage":None,"progress":0,"started_at":None,"updated_at":_now_iso()}

def _write_status(d):
    d["updated_at"] = _now_iso()
    STATUS_PATH.write_text(json.dumps(d, indent=2))

def _set_state(state:str, stage:str|None, progress:int):
    st = _read_status()
    st["state"] = state
    st["stage"] = stage
    st["progress"] = progress
    if state == "running" and st.get("started_at") is None:
        st["started_at"] = _now_iso()
    if state in ("idle","error","done"):
        # keep started_at for reference; add finished_at on done/error
        st["finished_at"] = _now_iso()
    _write_status(st)

def run_stage(cmd_args:list[str], stage_name:str):
    _set_state("running", stage_name, progress=0)
    try:
        subprocess.run([PY, *cmd_args], cwd=str(ROOT), check=True)
        _set_state("running", stage_name, progress=100)
    except subprocess.CalledProcessError as e:
        st = _read_status()
        st["state"] = "error"
        st["error"] = f"Stage '{stage_name}' failed: {e}"
        _write_status(st)
        raise

# Public helpers used by the UI (one-stage calls)
def run_fetch():        run_stage(STAGES[0][1], STAGES[0][0])
def run_features():     run_stage(STAGES[1][1], STAGES[1][0])
def run_optimize():     run_stage(STAGES[2][1], STAGES[2][0])
def run_walkforward():  run_stage(STAGES[3][1], STAGES[3][0])
def run_stress():       run_stage(STAGES[4][1], STAGES[4][0])
def run_backtest():     run_stage(STAGES[5][1], STAGES[5][0])
def run_attribution():  run_stage(STAGES[6][1], STAGES[6][0])

# Full pipeline
def run_all():
    total = len(STAGES)
    _set_state("running", None, progress=0)
    for i, (name, args) in enumerate(STAGES, start=1):
        # mark which stage is running and approximate progress
        pct = int((i-1)/total * 100)
        _set_state("running", name, pct)
        subprocess.run([PY, *args], cwd=str(ROOT), check=True)
        _set_state("running", name, int(i/total*100))
    _set_state("done", None, 100)