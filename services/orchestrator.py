# services/orchestrator.py
from pathlib import Path
import traceback

# services/orchestrator.py
import streamlit as st

def run_stage(func, label):
    """Run a pipeline stage with a progress message."""
    try:
        with st.spinner(f"Running {label}..."):
            func()
        st.success(f"{label} completed successfully!")
    except Exception as e:
        st.error(f"Error running {label}: {e}")
        raise

def _cfg_path() -> str:
    return str(Path("configs/default.yaml").resolve())

def run_fetch():      # expose as tiny wrappers so Streamlit can call
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pipeline.fetch_market", "--config", _cfg_path()])

def run_features():
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pipeline.build_features", "--config", _cfg_path()])

def run_optimize():
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pipeline.optimize", "--config", _cfg_path()])

def run_walkforward():
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pipeline.walkforward", "--config", _cfg_path()])

def run_stress():
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pipeline.stress", "--config", _cfg_path()])

def run_backtest():
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pipeline.backtest", "--config", _cfg_path()])

def run_attribution():
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pipeline.attribution", "--config", _cfg_path()])

def run_all():
    run_fetch(); run_features(); run_optimize(); run_walkforward(); run_stress(); run_backtest(); run_attribution()