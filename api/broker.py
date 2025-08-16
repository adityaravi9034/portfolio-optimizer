# api/broker.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from datetime import datetime

router = APIRouter(prefix="/broker", tags=["broker"])

# --- super-simple in-memory state (resets on server restart) ---
POSITIONS: Dict[str, float] = {}     # {asset: shares}
ORDERS: List[Dict[str, Any]] = []    # list of executed orders

class BlotterRow(BaseModel):
    asset: str
    approx_shares: float
    dollars: float | None = None
    last_price: float | None = None
    delta_weight: float | None = None

class ExecuteIn(BaseModel):
    orders: List[BlotterRow]

@router.get("/positions")
def get_positions():
    # return as a flat list for the UI
    out = [{"asset": a, "shares": float(sh)} for a, sh in POSITIONS.items()]
    return {"ok": True, "items": out}

@router.get("/orders")
def get_orders():
    return {"ok": True, "items": ORDERS}

@router.post("/execute")
def execute_orders(payload: ExecuteIn):
    if not payload.orders:
        raise HTTPException(status_code=400, detail="No orders provided")
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    applied = []
    for row in payload.orders:
        sh = float(row.approx_shares or 0.0)
        if abs(sh) < 1e-9:
            continue
        # update positions
        POSITIONS[row.asset] = POSITIONS.get(row.asset, 0.0) + sh
        # log order
        rec = {
            "ts": ts,
            "asset": row.asset,
            "approx_shares": sh,
            "dollars": row.dollars,
            "last_price": row.last_price,
            "delta_weight": row.delta_weight,
        }
        ORDERS.append(rec)
        applied.append(rec)
    return {"ok": True, "applied": applied, "positions": POSITIONS}

@router.post("/reset")
def reset_broker_state():
    POSITIONS.clear()
    ORDERS.clear()
    return {"ok": True}