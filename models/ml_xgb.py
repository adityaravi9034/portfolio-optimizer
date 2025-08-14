import numpy as np

def _train_xgb(X, y, params):
    import xgboost as xgb
    model = xgb.XGBRegressor(
        n_estimators=params.get("n_estimators", 400),
        max_depth=params.get("max_depth", 5),
        learning_rate=params.get("learning_rate", 0.05),
        subsample=params.get("subsample", 0.9),
        colsample_bytree=params.get("colsample_bytree", 0.9),
        random_state=params.get("random_state", 42),
    )
    model.fit(X, y)
    return model

def _train_lgbm(X, y, params):
    try:
        import lightgbm as lgb
    except Exception:
        return None
    model = lgb.LGBMRegressor(
        n_estimators=params.get("n_estimators", 600),
        num_leaves=params.get("num_leaves", 31),
        learning_rate=params.get("learning_rate", 0.03),
        feature_fraction=params.get("feature_fraction", 0.8),
        bagging_fraction=params.get("bagging_fraction", 0.8),
        bagging_freq=params.get("bagging_freq", 3),
        random_state=params.get("random_state", 42),
    )
    model.fit(X, y)
    return model

def _train_rf(X, y, params):
    from sklearn.ensemble import RandomForestRegressor as RF
    model = RF(
        n_estimators=params.get("n_estimators", 400),
        max_depth=params.get("max_depth", 8),
        min_samples_leaf=params.get("min_samples_leaf", 3),
        random_state=params.get("random_state", 42),
        n_jobs=-1,
    )
    model.fit(X, y)
    return model

def train_ml(X_train, y_train, cfg_ml):
    choice = (cfg_ml or {}).get("model", "xgboost").lower()
    try:
        if choice == "xgboost":
            return _train_xgb(X_train, y_train, cfg_ml.get("xgb", {}))
        if choice == "lightgbm":
            m = _train_lgbm(X_train, y_train, cfg_ml.get("lgbm", {}))
            if m is not None:
                return m
        # fallback
        return _train_rf(X_train, y_train, cfg_ml.get("rf", {}))
    except Exception:
        return _train_rf(X_train, y_train, cfg_ml.get("rf", {}))

def preds_to_weights(preds, cap=0.30, cash=0.02):
    ranks = preds.argsort().argsort()
    s = np.exp(ranks/3.0)
    w = s / s.sum()
    w = np.minimum(w, cap)
    w = w / w.sum() * (1.0 - cash)
    return w
