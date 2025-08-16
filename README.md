# 📈 Portfolio Optimizer with Machine Learning  

An end-to-end **portfolio optimization framework** that combines traditional quantitative finance methods with **machine learning–based return prediction**. The system is designed to support **research, backtesting, and live experimentation** with flexible configuration and a modular architecture.  

[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-312/)  
[![License](https://img.shields.io/badge/license-Proprietary-red.svg)]() 
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()  
[![Contributions](https://img.shields.io/badge/contributions-welcome-orange.svg)]()  

---

## 🚀 Project Goals  
- Provide an **end-to-end workflow**: from raw data → feature engineering → ML predictions → optimization → dashboard visualization.  
- Combine **traditional mean-variance optimization** with **machine learning forecasts** to improve allocation decisions.  
- Support **ensemble learning** with multiple ML models (linear, tree-based, deep learning).  
- Enable **transparent explainability** via feature importance tracking.  
- Expose a **web dashboard & REST API** for interactive exploration.  

---

## 🔧 Key Features  

### 📊 Data & Features  
- Automated pipeline to construct return & risk features:  
  - Mean (`feat_mu.csv`)  
  - Volatility (`feat_vol.csv`)  
  - Momentum (6M & 12M)  
  - Macro indicators (VIX, LFPR, etc.)  
- Configurable **lookback** and **stride** for rolling window construction.  

### 🤖 Machine Learning Forecasting  
- **Ensemble framework** with multiple models:  
  - Linear regression (baseline)  
  - Random Forests / Gradient Boosting  
  - XGBoost / LightGBM (with GPU/OpenMP support)  
  - Extensible to deep models (LSTMs/Transformers).  
- Model ensembling with **weighted averaging** of predictions.  
- Feature importance extraction (normalized, per-model, weight-adjusted).  
- Predictions & importances written to:  
  - `data/processed/ml_latest_preds.csv`  
  - `data/processed/ml_feature_importance.csv`  

### 📈 Optimization Engine  
- Mean-variance optimization with constraints.  
- Risk parity & custom allocation strategies.  
- Integrated with ML forecasts for **predictive optimization**.  

### 🖥️ Web Dashboard & API  
- Interactive dashboard (`app/dashboard.py`) for insights.  
- REST endpoints:  
  - `GET /ml/preds` → Latest ML forecasts.  
  - `GET /ml/importance` → Feature importance rankings.  
  - `POST /run/optimize` → Trigger optimization pipeline.  

### 📑 Reports & Backtesting  
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
└── README.md            # Project documentation
```

---

## ⚙️ Installation  

```bash
git clone https://github.com/adityaravi9034/portfolio-optimizer.git
cd portfolio-optimizer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ▶️ Usage  

### 1. Build Features  
```bash
python -m pipeline.build_features --config configs/default.yaml
```

### 2. Run ML Forecasts  
```bash
python -m pipeline.ml_forecast --config configs/default.yaml
```

### 3. Run Optimization  
```bash
python -m pipeline.optimize --config configs/default.yaml
```

### 4. Start API Server  
```bash
uvicorn api.main:app --reload --port 8000
```

### 5. Launch Dashboard  
```bash
python app/dashboard.py
```

### 6. Example API Calls  
```bash
curl http://127.0.0.1:8000/ml/preds
curl http://127.0.0.1:8000/ml/importance
```

---

## 📊 Example Outputs  

**Predictions (`/ml/preds`)**  
| Asset   | Pred_21d | Rank |
|---------|----------|------|
| ETH-USD | 0.1976   | 0.04 |
| USO     | 0.0313   | 0.08 |
| BTC-USD | 0.0234   | 0.12 |

**Feature Importance (`/ml/importance`)**  
| Model | Feature | Importance |
|-------|---------|------------|
| xgb   | mu      | 0.21       |
| xgb   | vol     | 0.27       |
| xgb   | m6      | 0.24       |
| xgb   | m12     | 0.28       |

---

## ✅ Recent Enhancements  
- Added **LightGBM** and **XGBoost** models into ensemble forecasting.  
- Improved **feature importance extraction** across pipelines and tree models.  
- Integrated model **weights** into importance calculation.  
- Expanded ensemble architecture for more robust asset ranking.  
- Modularized ML forecasting pipeline (`pipeline/ml_forecast.py`).  

---

## 🛠️ Roadmap  
- [ ] Add deep learning models (LSTM, Transformers).  
- [ ] Expand risk parity and factor-model based portfolio optimization.  
- [ ] Improve visualization layer with dynamic risk-return plots.  
- [ ] Deploy full stack on cloud (AWS/GCP/Azure).  
- [ ] Add live data feeds for trading simulation.  

---

## 📌 Notes  
- Ensure **LightGBM** and **XGBoost** are installed with OpenMP support (`brew install libomp` on macOS).  
- Auto-generated files (`__pycache__`, `data/processed/`, `data/reports/`) should be excluded with `.gitignore`.  

---

## 📜 License  

This project is licensed under a **Proprietary License**.  
All rights are reserved by the author (Aditya Ravi).  

You may not use, copy, modify, distribute, or create derivative works of this project without explicit written permission from the author.  

For inquiries regarding licensing, collaborations, or commercial usage, please contact:  
📧 adityaravu@gmail.com  
