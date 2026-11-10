"""Collect metrics for every finished run of the 0/1 depth sweep (in parallel), then render all figures.

Usage: python run_all.py
Figures go to plots_01_depth_sweep/ at the repo root; per-run metrics to plotdata/ next to this script.
Runs whose plotdata already exists are skipped; delete plotdata/<run>.json to recollect one.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CSV = os.path.join(REPO, "experiments_configs", "runs_01_depth_sweep.csv")

cfg = pd.read_csv(CSV, dtype=str, keep_default_na=False)
names = [os.path.basename(p.strip()) for p in cfg["result_path"] if p.strip().startswith("experiments/")]
todo = [n for n in names if not os.path.exists(os.path.join(HERE, "plotdata", f"{n}.json"))]
print(f"{len(names)} finished runs in the csv, {len(todo)} to collect")


def collect(name):
    r = subprocess.run([sys.executable, os.path.join(HERE, "collect_run.py"), name], capture_output=True, text=True)
    print(f"{name[-8:]}: {'ok' if r.returncode == 0 else 'FAILED'}", flush=True)
    if r.returncode != 0:
        print(r.stderr[-2000:])
    return r.returncode


with ThreadPoolExecutor(max_workers=4) as pool:
    codes = list(pool.map(collect, todo))
subprocess.run([sys.executable, os.path.join(HERE, "make_plots.py")], check=True)
sys.exit(max(codes, default=0))
