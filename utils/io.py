from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
PROC = Path("data/processed")

def load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    try:
        df.index = pd.to_datetime(df.index, utc=False, errors="coerce")
        df = df[~df.index.isna()]
    except Exception:
        pass
    return df

def save_df(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
