# models/ml_models.py
from __future__ import annotations
from typing import Any, Dict
import numpy as np

# Optional deps
try:
    from xgboost import XGBRegressor  # type: ignore
except Exception:
    XGBRegressor = None

try:
    import lightgbm as lgb  # type: ignore
except Exception:
    lgb = None

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def train_single_model(name: str, X_train, y_train, ml_cfg: Dict[str, Any]):
    """
    Build & fit one regressor by name. Falls back gracefully if a lib isn't installed.
    Supported names: 'ridge', 'rf', 'gradient_boosting', 'xgb', 'lgbm'
    """
    name = name.lower()
    params = (ml_cfg.get(name) or {})  # section like ml.xgb, ml.rf, etc.

    if name == "ridge":
        alpha = float(params.get("alpha", 1.0))
        model = Ridge(alpha=alpha, random_state=params.get("random_state", 42))

    elif name == "rf":
        model = RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 500)),
            max_depth=params.get("max_depth", 10),
            min_samples_leaf=int(params.get("min_samples_leaf", 3)),
            random_state=int(params.get("random_state", 42)),
            n_jobs=-1,
        )

    elif name == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=int(params.get("n_estimators", 300)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            max_depth=int(params.get("max_depth", 3)),
            random_state=int(params.get("random_state", 42)),
        )

    elif name == "xgb":
        if XGBRegressor is None:
            # fallback to ridge if xgboost missing
            model = Ridge(alpha=1.0, random_state=42)
        else:
            model = XGBRegressor(
                n_estimators=int(params.get("n_estimators", 500)),
                max_depth=int(params.get("max_depth", 6)),
                learning_rate=float(params.get("learning_rate", 0.05)),
                subsample=float(params.get("subsample", 0.9)),
                colsample_bytree=float(params.get("colsample_bytree", 0.9)),
                random_state=int(params.get("random_state", 42)),
                n_jobs=-1,
                tree_method=params.get("tree_method", "hist"),
            )

    elif name == "lgbm":
        if lgb is None:
            model = Ridge(alpha=1.0, random_state=42)
        else:
            model = lgb.LGBMRegressor(
                n_estimators=int(params.get("n_estimators", 700)),
                num_leaves=int(params.get("num_leaves", 31)),
                learning_rate=float(params.get("learning_rate", 0.03)),
                feature_fraction=float(params.get("feature_fraction", 0.8)),
                bagging_fraction=float(params.get("bagging_fraction", 0.8)),
                bagging_freq=int(params.get("bagging_freq", 3)),
                random_state=int(params.get("random_state", 42)),
                n_jobs=-1,
            )
    else:
        # default safe fallback
        model = Ridge(alpha=1.0, random_state=42)

    model.fit(X_train, y_train)
    return model


def preds_to_weights(preds: np.ndarray, *, cap: float, cash: float) -> np.ndarray:
    """
    Convert predicted forward returns into long-only, capped weights with cash buffer.
    """
    preds = np.nan_to_num(preds, nan=0.0)
    preds[preds < 0] = 0.0  # long-only tilt
    if preds.sum() <= 0:
        # all zero/negatives -> equal weight on non-zero entries (rare)
        nz = np.ones_like(preds)
        w = nz / nz.sum()
    else:
        w = preds / preds.sum()

    # apply cap
    w = np.minimum(w, cap)
    # renormalize leftover after cap
    if w.sum() > 0:
        w = w / w.sum()

    # apply cash buffer
    w = (1.0 - cash) * w
    return w