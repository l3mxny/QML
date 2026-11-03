"""Evaluate decoder-comparison rows 2-7 in parallel, then rerun the shots training check after the cache fix."""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
env = {**os.environ, "PYTHONIOENCODING": "utf-8", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}

logs = {row: open(os.path.join(HERE, f"cmp_{row}.log"), "w", encoding="utf-8") for row in range(2, 8)}
processes = {row: subprocess.Popen([sys.executable, os.path.join(HERE, "collect_compare.py"), str(row)], stdout=log, stderr=subprocess.STDOUT, env=env)
             for row, log in logs.items()}
codes = {row: process.wait() for row, process in processes.items()}
print("compare exit codes", codes, flush=True)

with open(os.path.join(HERE, "nocache_after_fix.log"), "w", encoding="utf-8") as log:
    code = subprocess.run([sys.executable, os.path.join(HERE, "collect_nocache.py")], stdout=log, stderr=subprocess.STDOUT, env=env).returncode
print("shots after fix exit code", code, flush=True)
