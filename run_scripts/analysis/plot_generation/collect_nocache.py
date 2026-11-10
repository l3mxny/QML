"""Shots training with the repo as it is now, i.e., after removing the on-disk compilation cache."""
import os
import sys

import jax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_tests as ct  # noqa: E402

assert not jax.config.jax_compilation_cache_dir, "the handler must not enable the cache"
jax.config.update("jax_enable_x64", True)
os.chdir(ct.HERE)
ct.save_runs([ct.train_and_log("FRQI, 256 shots, after fix", 150, shots=256),
              ct.train_and_log("FRQI, 2048 shots, after fix", 150, shots=2048)], "D2")
print("done", flush=True)
