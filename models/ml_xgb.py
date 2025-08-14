import numpy as np
import xgboost as xgb

def train_xgb(X_train, y_train):
    model = xgb.XGBRegressor(n_estimators=300, max_depth=4,
                             learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, random_state=42)
    model.fit(X_train, y_train)
    return model

def preds_to_weights(preds, cap=0.30, cash=0.02):
    ranks = preds.argsort().argsort()
    s = np.exp(ranks/3.0)
    w = s / s.sum()
    w = np.minimum(w, cap)
    w = w / w.sum() * (1.0 - cash)
    return w