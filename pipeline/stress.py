import argparse, yaml
from pathlib import Path
import pandas as pd
from utils.io import RAW, PROC, load_df, save_df

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    prices = load_df(RAW/'prices.csv')
    out = []
    for sp in cfg['stress_periods']:
        seg = prices.loc[sp['start']:sp['end']]
        ret = (seg.pct_change().add(1).prod()-1).sort_values(ascending=False)
        out.append(pd.DataFrame({'period':sp['name'], 'asset':ret.index, 'return':ret.values}))
    stress = pd.concat(out, ignore_index=True)
    save_df(stress, PROC/'stress_summary.csv')
    print('Saved stress summary')