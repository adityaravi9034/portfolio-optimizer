import argparse, subprocess, sys

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    cmds = [
        ['python','pipeline/fetch_market.py','--config',args.config],
        ['python','pipeline/build_features.py','--config',args.config],
        ['python','pipeline/optimize.py','--config',args.config]
    ]
    for c in cmds:
        print('Running:', ' '.join(c));
        ret = subprocess.call(c)
        if ret!=0:
            sys.exit(ret)
    print('Daily pipeline complete.')