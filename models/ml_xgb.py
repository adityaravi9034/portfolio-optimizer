import numpy as np
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    from sklearn.ensemble import RandomForestRegressor as RF
    HAS_XGB = False

def train_xgb(X_train, y_train):
    if HAS_XGB:
        model = xgb.XGBRegressor(
            n_estimators=300, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=42
        )
        model.fit(X_train, y_train)
        return model
    else:
        model = RF(
            n_estimators=300, max_depth=6,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        return model

def preds_to_weights(preds, cap=0.30, cash=0.02):
    ranks = preds.argsort().argsort()
    s = np.exp(ranks/3.0)
    w = s / s.sum()
    w = np.minimum(w, cap)
    w = w / w.sum() * (1.0 - cash)
    return w
