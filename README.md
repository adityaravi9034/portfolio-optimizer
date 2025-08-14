# Multi-Asset Portfolio Optimizer (Free Tier)

An end-to-end, zero-cost system that fetches free market data, engineers features, optimizes portfolios via multiple methods (MPT, risk parity, factor, ML/XGBoost), backtests with costs and stress tests, and deploys an interactive dashboard.

## Quickstart
1. Clone repo and install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run daily pipeline locally:
   ```bash
   python scripts/run_daily.py --config configs/default.yaml
   ```
3. Launch dashboard:
   ```bash
   streamlit run app/dashboard.py
   ```

## Deploy (Free)
- Streamlit Cloud: Connect this repo, set entry to `app/dashboard.py`.
- Hugging Face Spaces: Create a Streamlit Space and point to this repo.
