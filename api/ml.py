# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
# api/ml.py
from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pathlib import Path
import pandas as pd

router = APIRouter(prefix="/ml", tags=["ml"])

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

def _safe_read_csv(p: Path, **kw):
    if not p.exists() or p.stat().st_size == 0:
        return None
    return pd.read_csv(p, **kw)

@router.get("/preds")
def get_latest_preds():
    """
    Returns the latest cross-sectional ML predictions ranked, from
    data/processed/ml_latest_preds.csv written by pipeline/ml_forecast.py
    """
    df = _safe_read_csv(PROC / "ml_latest_preds.csv")
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No ML predictions available")
    # ensure expected columns
    cols = [c for c in df.columns if c.lower() in {"asset","pred_21d","rank"}]
    if not {"asset","pred_21d","rank"}.issubset(set([c.lower() for c in cols])):
        # try to coerce names
        df.columns = [c.lower() for c in df.columns]
    items = [
        {
            "asset": str(row.get("asset") or row.get("ticker")),
            "pred_21d": float(row.get("pred_21d")),
            "rank": float(row.get("rank")),
        }
        for _, row in df.iterrows()
        if pd.notna(row.get("asset")) and pd.notna(row.get("pred_21d"))
    ]
    return {"ok": True, "items": items}

@router.get("/importance")
def get_feature_importance():
    """
    Returns feature importance summary from data/processed/ml_feature_importance.csv
    """
    df = _safe_read_csv(PROC / "ml_feature_importance.csv")
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No ML importance file")
    # normalize column names
    df.columns = [c.lower() for c in df.columns]
    items = [
        {
            "model": str(row.get("model")),
            "weight": None if pd.isna(row.get("weight")) else float(row.get("weight")),
            "feature": None if pd.isna(row.get("feature")) else str(row.get("feature")),
            "importance": None if pd.isna(row.get("importance")) else float(row.get("importance")),
        }
        for _, row in df.iterrows()
    ]
    return {"ok": True, "items": items}