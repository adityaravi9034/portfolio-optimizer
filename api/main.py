# api/main.py
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# -----------------------------------------------------------------------------
# Status helpers (file-based, restart-safe)
# -----------------------------------------------------------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
STATUS_PATH = DATA_DIR / "run_status.json"
_status_lock = Lock()

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def read_status() -> Dict[str, Any]:
    if not STATUS_PATH.exists():
        return {
            "state": "idle",
            "stage": None,
            "started_at": None,
            "updated_at": _now_iso(),
            "finished_at": None,
            "progress": 0,
            "message": None,
            "error": None,
        }
    try:
        return json.loads(STATUS_PATH.read_text())
    except Exception:
        return {
            "state": "idle",
            "stage": None,
            "started_at": None,
            "updated_at": _now_iso(),
            "finished_at": None,
            "progress": 0,
            "message": "status parse error (reset next run)",
            "error": None,
        }

def write_status(**kwargs) -> Dict[str, Any]:
    with _status_lock:
        cur = read_status()
        cur.update(kwargs)
        cur["updated_at"] = _now_iso()
        STATUS_PATH.write_text(json.dumps(cur, indent=2))
        return cur

# Lightweight in-memory job registry (optional)
JOBS: Dict[str, Dict[str, Any]] = {}

# Expose status endpoints
router_status = APIRouter(prefix="/run", tags=["run"])

@router_status.get("/status")
def get_run_status():
    return {"ok": True, "status": read_status()}

@router_status.post("/status/reset")
def reset_run_status():
    write_status(
        state="idle", stage=None, started_at=None, finished_at=None,
        progress=0, message=None, error=None
    )
    return {"ok": True}

# -----------------------------------------------------------------------------
# Ensure project root importable
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

CFG_PATH = ROOT / "configs" / "default.yaml"
DATA_PROC = ROOT / "data" / "processed"

# Optional DB helpers
from services.db import get_conn, init_db  # type: ignore[import]

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="Portfolio Optimizer API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router_status)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _run(cmd: List[str]) -> None:
    """Run a subprocess and raise HTTPException on failure."""
    try:
        subprocess.check_call(cmd, cwd=str(ROOT))
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Subprocess failed: {e}")

def _py(*mod_and_args: str) -> List[str]:
    """Build a python -m command."""
    return [sys.executable, "-m", *mod_and_args]

def _csv_exists(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0

def _read_csv(path: Path, parse_dates: bool = False) -> List[Dict[str, Any]]:
    if not _csv_exists(path):
        return []
    if parse_dates:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = df.index.astype(str)
    else:
        df = pd.read_csv(path, index_col=0)
    rows: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        rows.append({"index": str(idx), **{c: (None if pd.isna(v) else v) for c, v in row.items()}})
    return rows

# -----------------------------------------------------------------------------
# Pydantic Schemas
# -----------------------------------------------------------------------------
class UniverseConstraints(BaseModel):
    universe: Optional[List[str]] = None
    max_weight: Optional[float] = None
    long_only: Optional[bool] = None
    cash_buffer: Optional[float] = None

class RunResponse(BaseModel):
    ok: bool
    message: str

class WeightsResponse(BaseModel):
    ok: bool
    model_names: List[str]
    weights: List[Dict[str, Any]]

class MetricsResponse(BaseModel):
    ok: bool
    metrics: List[Dict[str, Any]]

class EquityResponse(BaseModel):
    ok: bool
    equity: List[Dict[str, Any]]

# -----------------------------------------------------------------------------
# Strategies Router (CRUD)
# -----------------------------------------------------------------------------
router_strats = APIRouter(prefix="/strategies", tags=["strategies"])

class StrategyIn(BaseModel):
    name: str
    universe: List[str]
    max_weight: float
    long_only: bool
    cash_buffer: float

@router_strats.get("")
def list_strategies():
    with get_conn() as con:
        rows = con.execute("""
            SELECT name, universe, max_weight, long_only, cash_buffer, created_at, updated_at
            FROM strategies
            ORDER BY updated_at DESC
        """).fetchall()
        out = []
        for r in rows:
            out.append({
                "name": r["name"],
                "universe": r["universe"].split(","),
                "max_weight": r["max_weight"],
                "long_only": bool(r["long_only"]),
                "cash_buffer": r["cash_buffer"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            })
        return {"ok": True, "items": out}

@router_strats.get("/{name}")
def get_strategy(name: str):
    with get_conn() as con:
        r = con.execute("SELECT * FROM strategies WHERE name = ?", (name,)).fetchone()
        if not r:
            raise HTTPException(status_code=404, detail="Not found")
        return {"ok": True, "item": {
            "name": r["name"],
            "universe": r["universe"].split(","),
            "max_weight": r["max_weight"],
            "long_only": bool(r["long_only"]),
            "cash_buffer": r["cash_buffer"],
        }}

@router_strats.post("")
def upsert_strategy(s: StrategyIn):
    with get_conn() as con:
        con.execute("""
            INSERT INTO strategies(name, universe, max_weight, long_only, cash_buffer)
            VALUES(?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET
              universe=excluded.universe,
              max_weight=excluded.max_weight,
              long_only=excluded.long_only,
              cash_buffer=excluded.cash_buffer,
              updated_at=CURRENT_TIMESTAMP
        """, (s.name, ",".join(s.universe), s.max_weight, int(s.long_only), s.cash_buffer))
    return {"ok": True}

@router_strats.delete("/{name}")
def delete_strategy(name: str):
    with get_conn() as con:
        con.execute("DELETE FROM strategies WHERE name = ?", (name,))
    return {"ok": True}

app.include_router(router_strats)

# -----------------------------------------------------------------------------
# Stage definitions & runners
# -----------------------------------------------------------------------------
# List of pipeline stages: (friendly_name, python_module)
STAGES: List[Tuple[str, str]] = [
    ("fetch",       "pipeline.fetch_market"),
    ("features",    "pipeline.build_features"),
    ("optimize",    "pipeline.optimize"),
    ("walkforward", "pipeline.walkforward"),
    ("stress",      "pipeline.stress"),
    ("backtest",    "pipeline.backtest"),
    ("attribution", "pipeline.attribution"),
]

def _run_stage(stage_name: str, mod: str, idx: int, total: int) -> None:
    pct = int(round((idx / total) * 100))
    write_status(state="running", stage=stage_name, progress=pct, message=None, error=None)
    _run(_py(mod, "--config", str(CFG_PATH)))

def _run_all_background(job_id: str) -> None:
    try:
        total = len(STAGES)
        write_status(state="running", stage="starting", started_at=_now_iso(),
                     progress=0, message="pipeline starting", error=None)
        for i, (name, mod) in enumerate(STAGES, start=1):
            # mark before starting stage
            write_status(state="running", stage=name, progress=int((i - 1) * 100 / total))
            _run_stage(name, mod, i - 1, total)
            # mark after finishing stage
            write_status(state="running", stage=name, progress=int(i * 100 / total))
        write_status(state="done", stage="complete", progress=100, finished_at=_now_iso(),
                     message="all stages complete", error=None)
        JOBS[job_id] = {"state": "done", "error": None}
    except Exception as e:
        write_status(state="error", stage="failed", progress=0, finished_at=_now_iso(),
                     message=str(e), error=str(e))
        JOBS[job_id] = {"state": "error", "error": str(e)}

# -----------------------------------------------------------------------------
# Lifecycle
# -----------------------------------------------------------------------------
@app.on_event("startup")
def _on_startup():
    init_db()
    write_status(
        state="idle", stage=None, started_at=None, finished_at=None,
        progress=0, message=None, error=None
    )

# -----------------------------------------------------------------------------
# Basic endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"ok": True, "msg": "alive"}

@app.post("/config", response_model=RunResponse)
def update_config(cfg: UniverseConstraints):
    import yaml
    if not CFG_PATH.exists():
        raise HTTPException(status_code=400, detail="Config file not found.")
    data = yaml.safe_load(CFG_PATH.read_text())
    data.setdefault("universe", {})
    data.setdefault("constraints", {})
    if cfg.universe is not None:
        data["universe"]["equities"] = cfg.universe
    if cfg.max_weight is not None:
        data["constraints"]["max_weight"] = float(cfg.max_weight)
    if cfg.long_only is not None:
        data["constraints"]["long_only"] = bool(cfg.long_only)
    if cfg.cash_buffer is not None:
        data["constraints"]["cash_buffer"] = float(cfg.cash_buffer)
    CFG_PATH.write_text(yaml.safe_dump(data, sort_keys=False))
    return {"ok": True, "message": "config updated"}

# -----------------------------------------------------------------------------
# Stage endpoints (single)
# -----------------------------------------------------------------------------
@app.post("/run/fetch", response_model=RunResponse)
def run_fetch():
    write_status(state="running", stage="fetch", progress=0, started_at=_now_iso(),
                 finished_at=None, message=None, error=None)
    try:
        _run(_py("pipeline.fetch_market", "--config", str(CFG_PATH)))
    except HTTPException as e:
        write_status(state="error", stage="fetch", message=str(e), error=str(e))
        raise
    write_status(state="idle", stage="fetch", progress=100, finished_at=_now_iso(),
                 message="fetch complete", error=None)
    return {"ok": True, "message": "fetch complete"}

@app.post("/run/features", response_model=RunResponse)
def run_features():
    write_status(state="running", stage="features", progress=0, started_at=_now_iso(),
                 finished_at=None, message=None, error=None)
    try:
        _run(_py("pipeline.build_features", "--config", str(CFG_PATH)))
    except HTTPException as e:
        write_status(state="error", stage="features", message=str(e), error=str(e))
        raise
    write_status(state="idle", stage="features", progress=100, finished_at=_now_iso(),
                 message="features complete", error=None)
    return {"ok": True, "message": "features complete"}

@app.post("/run/optimize", response_model=RunResponse)
def run_optimize():
    write_status(state="running", stage="optimize", progress=0, started_at=_now_iso(),
                 finished_at=None, message=None, error=None)
    try:
        _run(_py("pipeline.optimize", "--config", str(CFG_PATH)))
    except HTTPException as e:
        write_status(state="error", stage="optimize", message=str(e), error=str(e))
        raise
    write_status(state="idle", stage="optimize", progress=100, finished_at=_now_iso(),
                 message="optimize complete", error=None)
    return {"ok": True, "message": "optimize complete"}

@app.post("/run/walkforward", response_model=RunResponse)
def run_walkforward():
    write_status(state="running", stage="walkforward", progress=0, started_at=_now_iso(),
                 finished_at=None, message=None, error=None)
    try:
        _run(_py("pipeline.walkforward", "--config", str(CFG_PATH)))
    except HTTPException as e:
        write_status(state="error", stage="walkforward", message=str(e), error=str(e))
        raise
    write_status(state="idle", stage="walkforward", progress=100, finished_at=_now_iso(),
                 message="walkforward complete", error=None)
    return {"ok": True, "message": "walkforward complete"}

@app.post("/run/stress", response_model=RunResponse)
def run_stress():
    write_status(state="running", stage="stress", progress=0, started_at=_now_iso(),
                 finished_at=None, message=None, error=None)
    try:
        _run(_py("pipeline.stress", "--config", str(CFG_PATH)))
    except HTTPException as e:
        write_status(state="error", stage="stress", message=str(e), error=str(e))
        raise
    write_status(state="idle", stage="stress", progress=100, finished_at=_now_iso(),
                 message="stress complete", error=None)
    return {"ok": True, "message": "stress complete"}

@app.post("/run/backtest", response_model=RunResponse)
def run_backtest():
    write_status(state="running", stage="backtest", progress=0, started_at=_now_iso(),
                 finished_at=None, message=None, error=None)
    try:
        _run(_py("pipeline.backtest", "--config", str(CFG_PATH)))
    except HTTPException as e:
        write_status(state="error", stage="backtest", message=str(e), error=str(e))
        raise
    write_status(state="idle", stage="backtest", progress=100, finished_at=_now_iso(),
                 message="backtest complete", error=None)
    return {"ok": True, "message": "backtest complete"}

@app.post("/run/attribution", response_model=RunResponse)
def run_attr():
    write_status(state="running", stage="attribution", progress=0, started_at=_now_iso(),
                 finished_at=None, message=None, error=None)
    try:
        _run(_py("pipeline.attribution", "--config", str(CFG_PATH)))
    except HTTPException as e:
        write_status(state="error", stage="attribution", message=str(e), error=str(e))
        raise
    write_status(state="idle", stage="attribution", progress=100, finished_at=_now_iso(),
                 message="attribution complete", error=None)
    return {"ok": True, "message": "attribution complete"}

# -----------------------------------------------------------------------------
# Run all (synchronous) – used by cURL or CI
# -----------------------------------------------------------------------------
@app.post("/run/all", response_model=RunResponse)
def run_all():
    total = len(STAGES)
    write_status(
        state="running", stage="starting", progress=0,
        started_at=_now_iso(), finished_at=None,
        message="pipeline starting", error=None
    )
    try:
        for i, (name, mod) in enumerate(STAGES, start=1):
            _run_stage(name, mod, i - 1, total)  # progress before stage
        write_status(state="done", stage="complete", progress=100, finished_at=_now_iso(),
                     message="all stages complete", error=None)
        return {"ok": True, "message": "all stages complete"}
    except HTTPException as e:
        write_status(state="error", stage=name, progress=int(((i - 1) / total) * 100),
                     finished_at=_now_iso(), message=str(e), error=str(e))
        raise

# -----------------------------------------------------------------------------
# Run all (background queue) – UI should call this, then poll /run/status
# -----------------------------------------------------------------------------
@app.post("/run/queue")
def run_queue():
    import uuid
    job_id = uuid.uuid4().hex
    JOBS[job_id] = {"state": "queued", "error": None}
    write_status(state="queued", stage="waiting", started_at=_now_iso(),
                 progress=0, message=None, error=None)
    t = Thread(target=_run_all_background, args=(job_id,), daemon=True)
    t.start()
    return {"ok": True, "job_id": job_id}

@app.get("/run/job/{job_id}")
def job_status(job_id: str):
    return {"ok": True, "job": JOBS.get(job_id, {"state": "unknown"})}

# -----------------------------------------------------------------------------
# Data access endpoints for UI
# -----------------------------------------------------------------------------
@app.get("/weights", response_model=WeightsResponse)
def get_weights():
    weights_path = DATA_PROC / "latest_weights.csv"
    rows = _read_csv(weights_path, parse_dates=False)
    model_names = [k for k in rows[0].keys() if k != "index"] if rows else []
    return {"ok": True, "model_names": model_names, "weights": rows}

@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    p = DATA_PROC / "latest_metrics.csv"
    if not _csv_exists(p):
        return {"ok": True, "metrics": []}
    s = pd.read_csv(p, index_col=0, header=None).squeeze()
    rows = [{"metric": str(idx), "value": None if pd.isna(v) else float(v)} for idx, v in s.items()]
    return {"ok": True, "metrics": rows}

@app.get("/equity_curves", response_model=EquityResponse)
def get_equity():
    rows = _read_csv(DATA_PROC / "equity_curves.csv", parse_dates=True)
    return {"ok": True, "equity": rows}