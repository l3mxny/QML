"""Check that the two decoders of one seed started from the same weights, and that different seeds did not."""
import os
import pickle

import jax
import numpy as np
import pandas as pd

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
os.chdir(os.path.join(REPO, "apps", "logistics"))
rows = pd.read_csv(os.path.join(REPO, "experiments_configs", "runs_small_circuits.csv"), dtype=str, keep_default_na=False).iloc[2:8]


def weights(path):
    with open(f"{path}/parameters_training_iteration=0.pickle", "rb") as f:
        generator_weights = pickle.load(f)[0]
    return np.concatenate([np.ravel(np.asarray(leaf)) for leaf in jax.tree_util.tree_leaves(generator_weights)])


initial = {(r.decoding_scheme, r.random_seed): weights(r.result_path) for r in rows.itertuples()}
for seed in ["42", "43", "44"]:
    a, b = initial[("FRQI", seed)], initial[("FRQI_msb", seed)]
    print(seed, "FRQI vs FRQI_msb max weight difference", float(np.abs(a - b).max()), "weights", a.size)
print("seed 42 vs 43", float(np.abs(initial[("FRQI", "42")] - initial[("FRQI", "43")]).max()))
print("seed 42 vs 44", float(np.abs(initial[("FRQI", "42")] - initial[("FRQI", "44")]).max()))
print(rows[["result_path", "decoding_scheme", "random_seed", "circuit_depth", "noise_tuning"]].assign(result_path=rows.result_path.str[-8:]).to_string())
