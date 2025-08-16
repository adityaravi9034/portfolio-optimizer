# api/main.py
from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()  # will read .env file automatically
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Dict, List, Optional, Tuple
from apscheduler.schedulers.background import BackgroundScheduler
import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pandas as pd
from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.broker import router as broker_router

# -----------------------------------------------------------------------------
# Project root & common paths (define ONCE, at the top)
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

CFG_PATH   = ROOT / "configs" / "default.yaml"
DATA_DIR   = ROOT / "data"
DATA_PROC  = DATA_DIR / "processed"
STATUS_PATH = DATA_DIR / "run_status.json"
REPORT_DIR  = DATA_DIR / "reports"

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Status helpers (file-based, restart-safe)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Optional auth (gated). If you don't have api/auth.py, this won't break.
# -----------------------------------------------------------------------------
current_user_dep = None
try:
    from api.auth import router as auth_router, current_user  # type: ignore
    current_user_dep = current_user
except Exception:
    auth_router = None
    # fallback dependency that returns a dummy user
    def _anon_user():
        return {"id": 0, "email": "anon@example.com", "name": "Anonymous"}
    current_user_dep = _anon_user

# -----------------------------------------------------------------------------
# Ensure DB helpers are importable
# -----------------------------------------------------------------------------
from services.db import get_conn, init_db, init_comments  # type: ignore

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="Portfolio Optimizer API", version="0.5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_methods=["*"],
    allow_headers=["*"],
)

if auth_router is not None:
    app.include_router(auth_router)

# -----------------------------------------------------------------------------
# Helpers for subprocess & CSV I/O
# -----------------------------------------------------------------------------
def _run(cmd: List[str]) -> None:
    try:
        subprocess.check_call(cmd, cwd=str(ROOT))
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Subprocess failed: {e}")

def _py(*mod_and_args: str) -> List[str]:
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
# Status Router
# -----------------------------------------------------------------------------
router_status = APIRouter(prefix="/run", tags=["run"])

@router_status.get("/status")
def get_run_status():
    return {"ok": True, "status": read_status()}

@router_status.post("/status/reset")
def reset_run_status():
    write_status(state="idle", stage=None, started_at=None, finished_at=None, progress=0, message=None, error=None)
    return {"ok": True}

app.include_router(router_status)

# -----------------------------------------------------------------------------
# Reports Router
# -----------------------------------------------------------------------------
router_reports = APIRouter(prefix="/report", tags=["report"])

def _load_proc_csv(name: str, parse_dates: bool = False):
    p = DATA_PROC / name
    if not p.exists() or p.stat().st_size == 0:
        return None
    df = pd.read_csv(p, index_col=0, parse_dates=parse_dates)
    return df

def _render_daily_html() -> str:
    eq  = _load_proc_csv("equity_curves.csv", parse_dates=True)
    kpi = _load_proc_csv("metrics_by_model.csv")
    w   = _load_proc_csv("latest_weights.csv")

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    html = [f"<h2>Daily Portfolio Report</h2><p><i>Generated {now}</i></p>"]

    if kpi is not None and not kpi.empty:
        html.append("<h3>Per-Model KPIs</h3>")
        html.append(kpi.to_html(float_format=lambda x: f"{x:0.4f}", border=0))
    else:
        html.append("<p><i>No KPIs available.</i></p>")

    if eq is not None and not eq.empty:
        last = eq.tail(1).T
        last.columns = ["Equity (x)"]
        html.append("<h3>Equity (Last Point)</h3>")
        html.append(last.to_html(float_format=lambda x: f"{x:0.3f}", border=0))
    if w is not None and not w.empty:
        html.append("<h3>Latest Weights</h3>")
        html.append(w.to_html(float_format=lambda x: f"{x:0.2%}", border=0))

    return "\n".join(html)

import smtplib, ssl
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv
load_dotenv()

# --- email helpers (optional) ---
import os, ssl, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

def _send_report_email(html_path: Path) -> None:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASS")
    to   = os.getenv("EMAIL_TO")

    # Silently skip if not configured
    if not (host and user and pwd and to):
        return

    html = html_path.read_text(encoding="utf-8")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Daily Portfolio Report — {datetime.utcnow().strftime('%Y-%m-%d')}"
    msg["From"] = user
    msg["To"] = to
    msg.attach(MIMEText(html, "html"))

    ctx = ssl.create_default_context()
    with smtplib.SMTP(host, port) as s:
        s.starttls(context=ctx)
        s.login(user, pwd)
        s.sendmail(user, [to], msg.as_string())



@router_reports.post("/daily")
def generate_daily_report():
    html = _render_daily_html()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = REPORT_DIR / f"report_{ts}.html"
    out.write_text(html, encoding="utf-8")
    return {"ok": True, "path": str(out.relative_to(ROOT))}

app.include_router(router_reports)

_scheduler: BackgroundScheduler | None = None

def _job_generate_daily_report():
    try:
        html = _render_daily_html()
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out = REPORT_DIR / f"report_{ts}.html"
        out.write_text(html, encoding="utf-8")
        write_status(message=f"Daily report generated: {out.name}")
    except Exception as e:
        write_status(state="error", stage="report", message=f"Report job failed: {e}")

    out = REPORT_DIR / f"report_{ts}.html"
    out.write_text(html, encoding="utf-8")
    _send_report_email(out)


@app.post("/email/test")
def send_test_email():
    try:
        tmp_path = REPORT_DIR / "test_email.html"
        tmp_path.write_text("<h2>✅ Test email from Portfolio Optimizer</h2>", encoding="utf-8")
        _send_report_email(tmp_path)
        return {"ok": True, "msg": "Test email sent"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.on_event("startup")
def _on_startup():
    from services.db import init_db, init_comments
    init_db()
    init_comments()
    write_status(state="idle", stage=None, started_at=None, finished_at=None, progress=0, message=None)

    global _scheduler
    _scheduler = BackgroundScheduler(timezone="UTC")
    # run every weekday at 21:00 UTC (adjust as you like)
    _scheduler.add_job(_job_generate_daily_report, "cron", day_of_week="mon-fri", hour=21, minute=0, id="daily_report")
    _scheduler.start()

@app.on_event("shutdown")
def _on_shutdown():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)

# -----------------------------------------------------------------------------
# Comments Router
# -----------------------------------------------------------------------------
# ---------------- Comments ----------------
router_comments = APIRouter(prefix="/comments", tags=["comments"])

class CommentIn(BaseModel):
    strategy: str
    author: str
    text: str

@router_comments.get("/{strategy}")
def get_comments(strategy: str, limit: int = 50):
    from services.db import list_comments
    return {"ok": True, "items": list_comments(strategy, limit)}

@router_comments.post("")
def post_comment(c: CommentIn):
    from services.db import add_comment
    try:
        add_comment(c.strategy, c.author, c.text)
        return {"ok": True}
    except ValueError as e:
        # bad input -> 400
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # unexpected DB error -> 500 with short message
        raise HTTPException(status_code=500, detail=f"comment insert failed: {e}")


app.include_router(router_comments)

# -----------------------------------------------------------------------------
# Insights Router
# -----------------------------------------------------------------------------
router_insights = APIRouter(prefix="/insights", tags=["insights"])

@router_insights.get("")
def get_insights():
    bullets = []
    try:
        kpi = pd.read_csv(DATA_PROC / "metrics_by_model.csv", index_col=0)
        if not kpi.empty:
            best = (kpi["Sharpe"].astype(float)).idxmax()
            bullets.append(f"Top Sharpe strategy: **{best}** (Sharpe {kpi.loc[best,'Sharpe']:.2f}).")
            # note: MaxDD is negative; 'least negative' is best
            dd_best = kpi["MaxDD"].astype(float).idxmax()
            bullets.append(f"Best max drawdown: **{dd_best}** (MaxDD {kpi.loc[dd_best,'MaxDD']:.2%}).")
    except Exception:
        pass

    try:
        eq = pd.read_csv(DATA_PROC / "equity_curves.csv", index_col=0, parse_dates=True)
        if not eq.empty:
            m = eq.columns[0]
            r = eq[m].pct_change().dropna()
            if len(r) > 20:
                last_20 = (1 + r.tail(20)).prod() - 1
                bullets.append(f"Last 1-month move (first model): **{last_20:.2%}**.")
    except Exception:
        pass

    if not bullets:
        bullets = ["Not enough data to generate insights yet."]
    return {"ok": True, "items": bullets[:5]}

app.include_router(router_insights)

# -----------------------------------------------------------------------------
# Schemas
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

# =========================
# Strategies Router (NO AUTH)
# =========================
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

router_strats = APIRouter(prefix="/strategies", tags=["strategies"])

class StrategyIn(BaseModel):
    name: str
    universe: List[str]
    max_weight: float
    long_only: bool
    cash_buffer: float

# ---------------- Reports ----------------
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse
from datetime import datetime
from pathlib import Path
import pandas as pd

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Portfolio Optimizer API", version="0.4.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# --- define the strategies router ONCE ---
router_strats = APIRouter(prefix="/strategies", tags=["strategies"])

@router_strats.get("")
def list_strategies():
    with get_conn() as con:
        rows = con.execute("""
            SELECT name, universe, max_weight, long_only, cash_buffer, created_at, updated_at
            FROM strategies
            ORDER BY updated_at DESC
        """).fetchall()
        return {"ok": True, "items": [
            {
                "name": r["name"],
                "universe": r["universe"].split(","),
                "max_weight": r["max_weight"],
                "long_only": bool(r["long_only"]),
                "cash_buffer": r["cash_buffer"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            } for r in rows
        ]}

# ... (other endpoints/routers) ...

# --- include routers ONCE (after they’re defined) ---
app.include_router(router_status)   # if you have it
app.include_router(router_strats)   # <- THIS ONE
app.include_router(router_reports)  # if you added reports
REPORT_DIR = (ROOT / "data" / "reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

def _load_proc_csv(name: str, parse_dates: bool = False):
    p = ROOT / "data" / "processed" / name
    if not p.exists() or p.stat().st_size == 0:
        return None
    return pd.read_csv(p, index_col=0, parse_dates=parse_dates)

def _render_daily_html() -> str:
    eq  = _load_proc_csv("equity_curves.csv", parse_dates=True)
    kpi = _load_proc_csv("metrics_by_model.csv")
    w   = _load_proc_csv("latest_weights.csv")

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        f"<h2>Daily Portfolio Report</h2>",
        f"<p><i>Generated {now}</i></p>",
    ]

    if kpi is not None and not kpi.empty:
        parts.append("<h3>Per-Model KPIs</h3>")
        parts.append(kpi.to_html(float_format=lambda x: f"{x:0.4f}", border=0))
    else:
        parts.append("<p><i>No KPIs available.</i></p>")

    if eq is not None and not eq.empty:
        last = eq.tail(1).T
        last.columns = ["Equity (x)"]
        parts.append("<h3>Equity (last point)</h3>")
        parts.append(last.to_html(float_format=lambda x: f"{x:0.3f}", border=0))

    if w is not None and not w.empty:
        parts.append("<h3>Latest Weights</h3>")
        parts.append(w.to_html(float_format=lambda x: f"{x:0.2%}", border=0))

    return "\n".join(parts)

def _safe_report_path(rel: str) -> Path:
    """
    Resolve a user-supplied relative path under REPORT_DIR safely.
    """
    # Allow either "data/reports/<file>.html" or just "<file>.html"
    if rel.startswith("data/reports/"):
        rel = rel[len("data/reports/"):]
    rel = rel.lstrip("/")

    p = (REPORT_DIR / rel).resolve()
    root_reports = REPORT_DIR.resolve()
    if not str(p).startswith(str(root_reports)):
        raise HTTPException(status_code=404, detail="invalid path")
    if not p.exists() or p.suffix.lower() != ".html":
        raise HTTPException(status_code=404, detail="not found")
    return p

@router_reports.post("/daily")
def generate_daily_report():
    """Render an HTML report and store it; return its relative path."""
    html = _render_daily_html()
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = REPORT_DIR / f"report_{ts}.html"
    out.write_text(html, encoding="utf-8")
    return {"ok": True, "path": str(out.relative_to(ROOT))}

@router_reports.get("/list")
def list_reports():
    """JSON list of report paths (relative to repo root), newest first."""
    items = sorted(
        [p for p in REPORT_DIR.glob("*.html") if p.is_file()],
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    rels = [str(p.relative_to(ROOT)) for p in items]  # e.g. "data/reports/report_20250816_154512.html"
    return {"ok": True, "items": rels}

@router_reports.get("/get/{relpath:path}", response_class=HTMLResponse)
def get_report(relpath: str):
    """Serve the HTML for a given report path."""
    p = _safe_report_path(relpath)
    return HTMLResponse(p.read_text(encoding="utf-8"))

from api import ml as ml_router
app.include_router(ml_router.router)
app.include_router(router_strats)    # /strategies
app.include_router(router_reports)   # /report/*# -----------------------------------------------------------------------------
# Stage definitions & runners
# -----------------------------------------------------------------------------
# api/main.py
STAGES = [
    ("fetch",       "pipeline.fetch_market"),
    ("features",    "pipeline.build_features"),
    ("optimize",    "pipeline.optimize"),     # ML happens here now
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
            write_status(state="running", stage=name, progress=int((i - 1) * 100 / total))
            _run_stage(name, mod, i - 1, total)
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
    init_comments()
    write_status(state="idle", stage=None, started_at=None, finished_at=None, progress=0, message=None, error=None)

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
    data = yaml.safe_load(CFG_PATH.read_text()) or {}
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
# Run all (sync) — handy for cURL/CI
# -----------------------------------------------------------------------------
@app.post("/run/all", response_model=RunResponse)
def run_all():
    total = len(STAGES)
    write_status(state="running", stage="starting", progress=0,
                 started_at=_now_iso(), finished_at=None,
                 message="pipeline starting", error=None)
    try:
        for i, (name, mod) in enumerate(STAGES, start=1):
            _run_stage(name, mod, i - 1, total)
        write_status(state="done", stage="complete", progress=100,
                     finished_at=_now_iso(), message="all stages complete", error=None)
        return {"ok": True, "message": "all stages complete"}
    except HTTPException as e:
        write_status(state="error", stage=name, progress=int(((i - 1) / total) * 100),
                     finished_at=_now_iso(), message=str(e), error=str(e))
        raise

# -----------------------------------------------------------------------------
# Run all (background) — UI should call /run/queue then poll /run/status
# -----------------------------------------------------------------------------
JOBS: Dict[str, Dict[str, Any]] = {}

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
# Data access for UI
# -----------------------------------------------------------------------------
@app.get("/weights", response_model=WeightsResponse)
def get_weights():
    rows = _read_csv(DATA_PROC / "latest_weights.csv", parse_dates=False)
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


app.include_router(router_insights)
app.include_router(router_comments)

# api/main.py
from services.db import init_comments, add_comment, list_comments

@app.on_event("startup")
def _on_startup():
    init_db()
    init_comments()
    write_status(state="idle", stage=None, started_at=None, finished_at=None, progress=0, message=None)

# Router
router_comments = APIRouter(prefix="/comments", tags=["comments"])

class CommentIn(BaseModel):
    strategy: str
    author: str
    text: str

@router_comments.get("/{strategy}")
def get_comments(strategy: str, limit: int = 50):
    return {"ok": True, "items": list_comments(strategy, limit)}

@router_comments.post("")
def post_comment(c: CommentIn):
    try:
        add_comment(c.strategy, c.author, c.text)
        return {"ok": True}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    

# ---------------- Email helpers & routes ----------------
from fastapi import APIRouter

router_email = APIRouter(prefix="/email", tags=["email"])

def _send_email_html(subject: str, html: str) -> dict:
    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASS")
    to   = os.getenv("EMAIL_TO")

    if not (host and user and pwd and to):
        return {"ok": False, "reason": "SMTP env not fully configured"}

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.attach(MIMEText(html, "html"))

    ctx = ssl.create_default_context()
    with smtplib.SMTP(host, port) as s:
        s.starttls(context=ctx)
        s.login(user, pwd)
        s.sendmail(user, [to], msg.as_string())

    return {"ok": True}

@router_email.post("/test")
def send_test_email():
    """Sends a simple test email to EMAIL_TO to verify SMTP settings."""
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    html = f"""
    <h3>Portfolio Optimizer — SMTP Test</h3>
    <p>This is a test email sent at <b>{now}</b>.</p>
    <p>If you received this, your SMTP settings are working ✅.</p>
    """
    res = _send_email_html("Portfolio Optimizer — SMTP Test", html)
    if not res.get("ok"):
        raise HTTPException(status_code=400, detail=res.get("reason", "send failed"))
    return {"ok": True}

app.include_router(router_email)

router_versions = APIRouter(prefix="/versions", tags=["versions"])

class VersionIn(BaseModel):
    strategy: str
    author: Optional[str] = None
    note: Optional[str] = None
    yaml_text: str

@router_versions.post("")
def save_version(v: VersionIn):
    from services.db import add_version
    vid = add_version(v.strategy.strip(), v.yaml_text, v.author, v.note)
    return {"ok": True, "id": vid}

@router_versions.get("/{strategy}")
def list_versions_api(strategy: str, limit: int = 20):
    from services.db import list_versions
    items = list_versions(strategy.strip(), limit)
    return {"ok": True, "items": items}

@router_versions.get("/get/{version_id}")
def get_version(version_id: int):
    from services.db import get_version_yaml
    y = get_version_yaml(int(version_id))
    if not y:
        raise HTTPException(status_code=404, detail="version not found")
    return {"ok": True, "yaml": y}

app.include_router(router_versions)

@app.on_event("startup")
def _on_startup():
    from services.db import init_db, init_comments, init_versions
    init_db()
    init_comments()
    init_versions()
    # …existing status init…

app.include_router(broker_router)

@app.post("/run/ml", response_model=RunResponse)
def run_ml():
    write_status(state="running", stage="ml", progress=0, started_at=_now_iso(),
                 finished_at=None, message=None, error=None)
    try:
        _run(_py("pipeline.ml_forecast", "--config", str(CFG_PATH)))
    except HTTPException as e:
        write_status(state="error", stage="ml", message=str(e), error=str(e))
        raise
    write_status(state="idle", stage="ml", progress=100, finished_at=_now_iso(),
                 message="ml complete", error=None)
    return {"ok": True, "message": "ml complete"}