# pipeline/ml_forecast.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

TRADING_DAYS = 252


# -----------------------------
# IO helpers
# -----------------------------
def _rd_csv(p: Path, **kw) -> pd.DataFrame | None:
    if not p.exists() or p.stat().st_size == 0:
        return None
    kw.setdefault("index_col", 0)
    df = pd.read_csv(p, **kw)
    # Try parse index to dates
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    return df


def _load_cfg(cfg_path: Path) -> dict:
    cfg = yaml.safe_load(cfg_path.read_text())
    return cfg.get("ml", {})


# -----------------------------
# Modeling helpers
# -----------------------------
def _build_ensemble(mlcfg: dict):
    """Return list of (name, estimator, weight)."""
    out = []
    models_cfg = mlcfg.get("models", []) or []

    # Helper to pull sub-configs with defaults
    def get_subcfg(key, default=None):
        return mlcfg.get(key, {}) or (default or {})

    for m in models_cfg:
        name = (m.get("name") or "").lower().strip()
        w = float(m.get("weight", 1.0))

        if name == "ridge":
            ridge_cfg = get_subcfg("ridge", {"alpha": 1.0, "fit_intercept": True, "random_state": 42})
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            est = Pipeline([
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("ridge", Ridge(alpha=float(ridge_cfg.get("alpha", 1.0)),
                                fit_intercept=bool(ridge_cfg.get("fit_intercept", True)),
                                random_state=int(ridge_cfg.get("random_state", 42))))])

        elif name == "random_forest":
            rf_cfg = get_subcfg("random_forest", {"n_estimators": 500, "max_depth": 10, "min_samples_leaf": 3, "random_state": 42})
            from sklearn.ensemble import RandomForestRegressor
            est = RandomForestRegressor(
                n_estimators=int(rf_cfg.get("n_estimators", 500)),
                max_depth=rf_cfg.get("max_depth", 10),
                min_samples_leaf=int(rf_cfg.get("min_samples_leaf", 3)),
                random_state=int(rf_cfg.get("random_state", 42)),
                n_jobs=-1,
            )

        elif name == "gradient_boosting":
            gb_cfg = get_subcfg("gradient_boosting", {"n_estimators": 500, "learning_rate": 0.05, "max_depth": 3, "subsample": 0.9, "random_state": 42})
            from sklearn.ensemble import GradientBoostingRegressor
            est = GradientBoostingRegressor(
                n_estimators=int(gb_cfg.get("n_estimators", 500)),
                learning_rate=float(gb_cfg.get("learning_rate", 0.05)),
                max_depth=int(gb_cfg.get("max_depth", 3)),
                subsample=float(gb_cfg.get("subsample", 0.9)),
                random_state=int(gb_cfg.get("random_state", 42)),
            )

        elif name == "lgbm":
            # LightGBM (optional)
            try:
                import lightgbm as lgb
                lgb_cfg = get_subcfg("lgbm", {"n_estimators": 700, "num_leaves": 31, "learning_rate": 0.03,
                                              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 3,
                                              "random_state": 42})
                est = lgb.LGBMRegressor(
                    n_estimators=int(lgb_cfg.get("n_estimators", 700)),
                    num_leaves=int(lgb_cfg.get("num_leaves", 31)),
                    learning_rate=float(lgb_cfg.get("learning_rate", 0.03)),
                    feature_fraction=float(lgb_cfg.get("feature_fraction", 0.8)),
                    bagging_fraction=float(lgb_cfg.get("bagging_fraction", 0.8)),
                    bagging_freq=int(lgb_cfg.get("bagging_freq", 3)),
                    random_state=int(lgb_cfg.get("random_state", 42)),
                )
            except Exception as e:
                print(f"[ml] Skipping LGBM (lightgbm not installed?): {e}")
                continue

        elif name == "xgb":
            # XGBoost (optional)
            try:
                from xgboost import XGBRegressor
                xgb_cfg = get_subcfg("xgb", {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
                                             "subsample": 0.9, "colsample_bytree": 0.9, "random_state": 42})
                est = XGBRegressor(
                    n_estimators=int(xgb_cfg.get("n_estimators", 500)),
                    max_depth=int(xgb_cfg.get("max_depth", 6)),
                    learning_rate=float(xgb_cfg.get("learning_rate", 0.05)),
                    subsample=float(xgb_cfg.get("subsample", 0.9)),
                    colsample_bytree=float(xgb_cfg.get("colsample_bytree", 0.9)),
                    random_state=int(xgb_cfg.get("random_state", 42)),
                    n_jobs=-1,
                    objective="reg:squarederror",
                )
            except Exception as e:
                print(f"[ml] Skipping XGB (xgboost not installed?): {e}")
                continue

        elif name == "lstm":
            # Placeholder: add only if you’ve implemented sequence pipeline (see notes below)
            print("[ml] 'lstm' requested, but sequence pipeline not enabled in this script. Skipping.")
            continue

        else:
            # unknown model id; skip quietly
            continue

        out.append((name, est, w))

    # normalize weights
    sw = sum(w for _, _, w in out) or 1.0
    out = [(n, e, w / sw) for (n, e, w) in out]
    return out


def _stack_feature(df: pd.DataFrame, colname: str) -> pd.DataFrame:
    """Wide [date x asset] -> long indexed by (date, asset) with a single column=colname."""
    if df is None or df.empty:
        return pd.DataFrame(columns=[colname])
    s = df.stack().to_frame(colname)
    s.index.set_names(["date", "asset"], inplace=True)
    return s


def _make_targets(prices: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Future arithmetic returns over `horizon` days, aligned to t (predict t+h - t)."""
    fwd = prices.pct_change(periods=horizon).shift(-horizon)
    fwd.index = pd.to_datetime(fwd.index)
    return fwd


# -----------------------------
# Main
# -----------------------------
def main(cfg_path: Path):
    mlcfg = _load_cfg(cfg_path)
    if not mlcfg.get("enabled", True):
        print("[ml] disabled via config")
        return

    # ---- config values ----
    horizon = int(mlcfg.get("horizon_days", 21))
    lookback_days = int(mlcfg.get("lookback_days", 756))          # (not used in this simple last-date fit)
    retrain_every = int(mlcfg.get("retrain_every_days", 21))      # (not used in this simple last-date fit)
    feat_cols = mlcfg.get("feature_columns") or ["mu", "vol", "m6", "m12"]

    use_sep = bool(mlcfg.get("use_separate_feature_files", True))
    mu_path  = ROOT / mlcfg.get("mu_path",  "data/processed/feat_mu.csv")
    vol_path = ROOT / mlcfg.get("vol_path", "data/processed/feat_vol.csv")
    m6_path  = ROOT / mlcfg.get("m6_path",  "data/processed/feat_mom6.csv")
    m12_path = ROOT / mlcfg.get("m12_path", "data/processed/feat_mom12.csv")

    prices_path  = ROOT / mlcfg.get("prices_path", "data/raw/prices.csv")
    out_pred     = ROOT / mlcfg.get("out_pred_path", "data/processed/ml_latest_preds.csv")
    out_imp      = ROOT / mlcfg.get("out_imp_path",  "data/processed/ml_feature_importance.csv")

    # ---- load data ----
    prices = _rd_csv(prices_path)
    if prices is None or prices.empty:
        raise RuntimeError("prices.csv missing or empty")

    if use_sep:
        mu  = _rd_csv(mu_path)
        vol = _rd_csv(vol_path)
        m6  = _rd_csv(m6_path)
        m12 = _rd_csv(m12_path)
        if any(df is None or df.empty for df in [mu, vol, m6, m12]):
            raise RuntimeError("One or more feature CSVs missing (feat_mu/feat_vol/feat_mom6/feat_mom12)")

        # align by intersection (dates & columns)
        cols = list(set(mu.columns) & set(vol.columns) & set(m6.columns) & set(m12.columns) & set(prices.columns))
        if not cols:
            raise RuntimeError("No overlapping assets between features and prices")

        mu, vol, m6, m12, prices = [df[cols].copy() for df in (mu, vol, m6, m12, prices)]
        dates = mu.index.intersection(vol.index).intersection(m6.index).intersection(m12.index).intersection(prices.index)
        if dates.empty:
            raise RuntimeError("No overlapping dates between features and prices")
        mu, vol, m6, m12, prices = [df.loc[dates].copy() for df in (mu, vol, m6, m12, prices)]

        # build long feature matrix
        X_long = _stack_feature(mu, "mu") \
            .join(_stack_feature(vol, "vol"), how="inner") \
            .join(_stack_feature(m6,  "m6"),  how="inner") \
            .join(_stack_feature(m12, "m12"), how="inner")
        X_long = X_long.dropna()
    else:
        # keeping an alternative path (unused here)
        raise RuntimeError("This build expects use_separate_feature_files=true")

    # ---- make targets & align ----
    y_wide = _make_targets(prices, horizon=horizon)
    y_long = y_wide.stack().to_frame("y")
    y_long.index.set_names(["date", "asset"], inplace=True)

    XY = X_long.join(y_long, how="inner").dropna()
    if XY.empty:
        raise RuntimeError("No aligned rows between features and targets")

    # ---- simple scheme: train on all dates < last_date; predict on last_date ----
    last_date = XY.index.get_level_values(0).max()
    mask_train = XY.index.get_level_values(0) < last_date
    train = XY.loc[mask_train]

    # Robust slice for testX: handle both MultiIndex and an already-reduced single-level index
    try:
        testX = X_long.loc[(last_date, slice(None))]
    except Exception:
        # if selecting by last_date produces single-level index of assets
        testX = X_long.loc[last_date]
    testX = testX.dropna()

    if testX.empty or train.empty:
        raise RuntimeError("Insufficient rows for ML (train or test empty)")

    # subset to desired features
    keep = [c for c in feat_cols if c in train.columns]
    if not keep:
        raise RuntimeError("None of the requested feature_columns exist in stacked features")

    Xtr = train[keep].values
    ytr = train["y"].values
    Xte = testX[keep].values

    # assets for predictions
    if isinstance(testX.index, pd.MultiIndex):
        assets_pred = testX.index.get_level_values(-1).tolist()
    else:
        assets_pred = testX.index.tolist()

    # ---- build & fit ensemble ----
    models = _build_ensemble(mlcfg)
    if not models:
        # ensure at least ridge if not configured
        models = [("ridge", Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0, random_state=42))]), 1.0)]

    preds_sum = np.zeros(len(Xte), dtype=float)
    total_w = 0.0
    # collect crude feature importances if available
    feat_imps = []  # (model_name, feature, importance)

    for name, est, w in models:
        est.fit(Xtr, ytr)
        p = est.predict(Xte).astype(float)
        preds_sum += w * p
        total_w += w

        # crude importance: try attribute if available (tree models)
        # crude importance: normalize per-model; handle bare estimators and Pipelines
    try:
        fi = None

        # direct attribute (e.g., RandomForestRegressor, GradientBoostingRegressor)
        if hasattr(est, "feature_importances_"):
            fi = est.feature_importances_

        # pipeline case (e.g., Pipeline([..., ('model', LGBMRegressor(...))]))
        if fi is None and hasattr(est, "named_steps"):
            for step in est.named_steps.values():
                if hasattr(step, "feature_importances_"):
                    fi = step.feature_importances_
                    break

        if fi is not None:
            fi = np.asarray(fi, dtype=float)
            # guard against length mismatch (e.g., feature selection in the pipeline)
            if len(fi) == len(keep):
                s = fi.sum()
                if s > 0:
                    fi = fi / s  # normalize so importances are comparable across models
                for f, v in zip(keep, fi):
                    feat_imps.append((name, f, float(v), float(w)))  # include model weight
    except Exception:
        pass

    if total_w == 0:
        total_w = 1.0
    preds = preds_sum / total_w

    # ---- write outputs expected by API/UI ----
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    latest = pd.DataFrame({"asset": assets_pred, "pred_21d": preds})
    latest["rank"] = latest["pred_21d"].rank(ascending=False, method="first") / len(latest)
    latest.sort_values("pred_21d", ascending=False, inplace=True)
    latest.to_csv(out_pred, index=False)
    print(f"[ml] wrote {out_pred}")

    # Build a single importance DataFrame and write once
    if feat_imps:
        imp_df = pd.DataFrame(feat_imps, columns=["model", "feature", "importance", "model_weight"])
    else:
        # fallback meta if models don’t expose FI
        imp_df = pd.DataFrame({
            "model": [n for n, _, _ in models],
            "feature": [", ".join(keep)] * len(models),
            "importance": [1.0 / len(models)] * len(models),
            "model_weight": [w for _, _, w in models],
        })

    imp_df.to_csv(out_imp, index=False)
    print(f"[ml] wrote {out_imp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()
    main(Path(args.config))