# 📈 Portfolio Optimizer with Machine Learning  

An end-to-end **portfolio optimization framework** that combines traditional quantitative finance methods with **machine learning–based return prediction**. The system is designed to support **research, backtesting, and live experimentation** with flexible configuration and a modular architecture.  

---

## 🚀 Project Goals  
- Provide an **end-to-end workflow**: from raw data → feature engineering → ML predictions → optimization → dashboard visualization.  
- Combine **traditional mean-variance optimization** with **machine learning forecasts** to improve allocation decisions.  
- Support **ensemble learning** with multiple ML models (linear, tree-based, deep learning).  
- Enable **transparent explainability** via feature importance tracking.  
- Expose a **web dashboard & REST API** for interactive exploration.  

---

## 🔧 Key Features  

### 1. **Data & Features**  
- Built-in pipeline to construct return & risk features:  
  - Mean (`feat_mu.csv`)  
  - Volatility (`feat_vol.csv`)  
  - Momentum (6M & 12M)  
  - Macro indicators (VIX, LFPR, etc.)  
- Configurable **lookback** and **stride** for rolling window construction.  

### 2. **Machine Learning Integration**  
- Ensemble framework with multiple models:  
  - **Linear regression** (baseline)  
  - **Random Forests / Gradient Boosting**  
  - **XGBoost / LightGBM** (with GPU/OpenMP support)  
  - Easily extensible to LSTMs or other deep models.  
- Model ensembling with **weighted averaging** of predictions.  
- Feature importance extraction (normalized, per-model, weight-adjusted).  
- Predictions and importances written to:  
  - `data/processed/ml_latest_preds.csv`  
  - `data/processed/ml_feature_importance.csv`  

### 3. **Optimization Engine**  
- Mean-variance optimization with constraints.  
- Risk parity & custom allocation strategies.  
- Integrated with ML forecasts for **predictive optimization**.  

### 4. **Web Dashboard & API**  
- Interactive dashboard (`app/dashboard.py`) for insights.  
- REST endpoints for automation:  
  - `GET /ml/preds` → Latest ML forecasts.  
  - `GET /ml/importance` → Feature importance rankings.  
  - `POST /run/optimize` → Trigger optimization pipeline.  

### 5. **Reports & Backtesting**  
- Auto-generated HTML reports in `data/reports/`.  
- Track optimization runs via `data/run_status.json`.  
- Logging with run history for reproducibility.  

---

## 📂 Repository Structure  

```
portfolio-optimizer-full/
├── api/                 # REST API (FastAPI)
│   └── ml.py            # ML prediction endpoints
├── app/                 # Dashboard UI
│   └── components/ml.py # ML predictions/importance display
├── configs/             # YAML configs (lookback, stride, assets, etc.)
├── data/                
│   ├── processed/       # Generated ML features/predictions
│   └── reports/         # Auto-generated reports
├── models/              # ML model definitions
│   └── ml_models.py
├── pipeline/            # Data & ML pipelines
│   ├── build_features.py
│   ├── ml_forecast.py   # Ensemble training & prediction
│   └── optimize.py      # Portfolio optimization
```

---

## ⚙️ Usage  

### 1. Setup Environment  
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Build Features  
```bash
python -m pipeline.build_features --config configs/default.yaml
```

### 3. Run ML Forecasts  
```bash
python -m pipeline.ml_forecast --config configs/default.yaml
```

### 4. Trigger Optimization  
```bash
curl -X POST http://127.0.0.1:8000/run/optimize
```

### 5. Fetch Predictions & Importance  
```bash
curl http://127.0.0.1:8000/ml/preds
curl http://127.0.0.1:8000/ml/importance
```

---

## 📊 Example Outputs  

### Predictions (`/ml/preds`)  
| Asset   | Pred_21d | Rank |
|---------|----------|------|
| ETH-USD | 0.1976   | 0.04 |
| USO     | 0.0313   | 0.08 |
| BTC-USD | 0.0234   | 0.12 |

### Feature Importance (`/ml/importance`)  
| Model | Feature | Importance |
|-------|---------|------------|
| xgb   | mu      | 0.21       |
| xgb   | vol     | 0.27       |
| xgb   | m6      | 0.24       |
| xgb   | m12     | 0.28       |

---

## 🛠️ Next Steps  
- Add **deep learning (LSTM/Transformers)** into the ensemble.  
- Enhance **risk-adjusted metrics** in optimization.  
- Deploy **cloud-native version** (GCP / AWS).  
- Add **streaming data** for live trading simulation.  

---

## 📌 Notes  
- Make sure to install **LightGBM** and **XGBoost** with OpenMP enabled (`brew install libomp` on macOS).  
- Auto-generated files (`__pycache__`, `data/processed/`, `data/reports/`) should be excluded with `.gitignore`.  
