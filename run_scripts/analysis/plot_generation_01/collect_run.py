"""Per-checkpoint metrics of one run of the 0/1 depth sweep (experiments_configs/runs_01_depth_sweep.csv).

Usage: python collect_run.py <run folder name> [experiments root]
The experiments root defaults to apps/logistics/experiments. Writes plotdata/<run folder name>.json and .npz.
Every checkpoint samples from the same noise inputs, so differences between checkpoints are not sampling noise.
"""
import contextlib
import io
import json
import os
import pickle
import re
import sys
import warnings
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("PLOTDATA_DIR", os.path.join(HERE, "plotdata"))
os.makedirs(OUT, exist_ok=True)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LOGISTICS = os.path.join(REPO, "apps", "logistics")

import jax  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler  # noqa: E402

jax.config.update("jax_enable_x64", True)  # as in train_image_qgan.main
warnings.simplefilter("ignore", ConvergenceWarning)

RUN = sys.argv[1]
EXP_ROOT = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else os.path.join(LOGISTICS, "experiments")
DATA_SET = "mnist_0_1_4x4_N_12665_area"
WANTED = [0, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
N = 4000

data = np.load(os.path.join(LOGISTICS, "training_data", f"{DATA_SET}.npy")) / 255.
labels = np.load(os.path.join(LOGISTICS, "training_data", f"{DATA_SET}_labels.npy")).astype(int)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=0, stratify=labels)
real = x_test[:N]
# Real-data 0-vs-1 classifier: tells which digit a generated image looks like
digit_clf = MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True, max_iter=300, random_state=0).fit(x_train, y_train)


def c2st_auc(a, b, sigma=0.05, seed=0):
    """AUC of a classifier telling real from generated (0.5 = indistinguishable, 1.0 = trivially different).
    Pixel values are blurred slightly so the test judges image structure, not numerical precision."""
    x = np.vstack([a, b]) + np.random.default_rng(seed).normal(0, sigma, (len(a) + len(b), a.shape[1]))
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    x_a, x_b, y_a, y_b = train_test_split(x, y, test_size=0.3, random_state=seed, stratify=y)
    p = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(x_a, y_a).predict_proba(x_b)[:, 1]
    return float(roc_auc_score(y_b, p))


def nn_rmse(query, pool):
    d, _ = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(pool).kneighbors(query)
    return float(d.mean() / np.sqrt(query.shape[1]))


run_dir = os.path.join(EXP_ROOT, RUN)
saved = sorted(int(re.search(r"=(\d+)\.pickle$", p).group(1))
               for p in glob(os.path.join(run_dir, "parameters_training_iteration=*.pickle")))
epochs = sorted({e for e in WANTED if e in saved} | {saved[-1]})

model = ContinuousQGANModelHandler()
cwd = os.getcwd()
os.chdir(os.path.dirname(EXP_ROOT))  # reload looks for experiments/<name> relative to the working directory
with contextlib.redirect_stdout(io.StringIO()):
    model.reload(model_name=RUN, epoch=saved[-1])
os.chdir(cwd)
# min-max scale of the training data (its brightest pixel is 254/255), so samples are on the same pixel scale as data
assert np.allclose(np.asarray(model.reverse_lookup, dtype=float), [data.min(), data.max()]), model.reverse_lookup

meta = model.metadata
real_ref = dict(c2st_auc=c2st_auc(x_test[:N], x_train[:N]), share_1=float((y_test == 1).mean()),
                nn_rmse=nn_rmse(x_test[:2000], x_train[:N]))
rows, arrays = [], dict(real_mean_0=x_test[y_test == 0].mean(0), real_mean_1=x_test[y_test == 1].mean(0),
                        real_grid=np.r_[x_test[y_test == 0][:8], x_test[y_test == 1][:8]])
for epoch in epochs:
    with open(os.path.join(run_dir, f"parameters_training_iteration={epoch}.pickle"), "rb") as f:
        model.generator_weights, _ = pickle.load(f)
    model.random_key = jax.random.PRNGKey(2026)
    fake = np.asarray(model.sample(N)).reshape(N, -1)
    assert not np.isnan(fake).any(), f"NaN samples at {epoch}"
    pred = digit_clf.predict(fake)
    row = dict(epoch=epoch, c2st_auc=c2st_auc(real, fake), share_1=float((pred == 1).mean()),
               confidence=float(digit_clf.predict_proba(fake).max(1).mean()),
               mean_image_mae=float(np.abs(fake.mean(0) - real.mean(0)).mean()),
               fake_to_real_rmse=nn_rmse(fake[:2000], x_train[:N]), real_to_fake_rmse=nn_rmse(x_test[:2000], fake))
    for d in (0, 1):
        if (pred == d).any():
            row[f"digit_mae_{d}"] = float(np.abs(fake[pred == d].mean(0) - arrays[f"real_mean_{d}"]).mean())
    arrays[f"film_{epoch}"] = fake[:8]
    if epoch == epochs[-1]:
        arrays["grid"] = np.r_[fake[pred == 0][:8], fake[pred == 1][:8]]
        arrays["grid_labels"] = np.r_[np.zeros(min(8, (pred == 0).sum())), np.ones(min(8, (pred == 1).sum()))]
        for d in (0, 1):
            arrays[f"fake_mean_{d}"] = fake[pred == d].mean(0) if (pred == d).any() else np.full(16, np.nan)
    rows.append(row)
    print(RUN[-8:], epoch, {k: round(v, 3) for k, v in row.items() if k != "epoch"}, flush=True)

with open(os.path.join(OUT, f"{RUN}.json"), "w") as f:
    json.dump(dict(model=RUN, circuit_depth=list(meta["circuit_depth"]), random_seed=meta.get("random_seed"),
                   decoding=meta["decoding_scheme"], real_ref=real_ref, rows=rows), f, indent=1)
np.savez(os.path.join(OUT, f"{RUN}.npz"), **arrays)
