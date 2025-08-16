# Copyright (c) 2025 Aditya Ravi
# All rights reserved.
import argparse, subprocess, sys

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    c = ['python','pipeline/backtest.py','--config',args.config]
    print('Running:', ' '.join(c))
    sys.exit(subprocess.call(c))