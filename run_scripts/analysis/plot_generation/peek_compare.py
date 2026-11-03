"""Quick coverage check of the running decoder comparison at one checkpoint. Usage: python peek_compare.py <iteration>"""
import contextlib
import io
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(os.path.join(REPO, "apps", "logistics"))

import glob  # noqa: E402
import json  # noqa: E402

import jax  # noqa: E402
import numpy as np  # noqa: E402

from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler  # noqa: E402

jax.config.update("jax_enable_x64", True)
epoch = int(sys.argv[1])
with open(os.path.join(HERE, "plotdata", "refs.pkl"), "rb") as f:
    clf_real, refs = pickle.load(f)
data = np.load("./training_data/mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area.npy") / 255.
labels = np.load("./training_data/mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area_labels.npy")
real_shares, real_mean = np.bincount(labels, minlength=10) / len(labels), data.mean(0)
for path in sorted(glob.glob("experiments/*/meta.json")):
    meta = json.load(open(path))
    if meta.get("circuit_depth") != [2, 1] or "4x4_N_60000_area" not in meta["data_set"]:
        continue
    name = meta["model_name"]
    model = ContinuousQGANModelHandler()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model.reload(model_name=name, epoch=epoch)
    model.random_key = jax.random.PRNGKey(2026)
    fake = np.asarray(model.sample(5000)).reshape(5000, -1)
    pred = clf_real.predict(fake)
    shares = np.bincount(pred, minlength=10) / len(pred)
    print(f"{name[-8:]} {meta['decoding_scheme']:9s} it={epoch} digits={int((shares >= 0.5 * real_shares).sum())} "
          f"tv={0.5 * np.abs(shares - real_shares).sum():.3f} conf={clf_real.predict_proba(fake).max(1).mean():.3f} "
          f"mean_image_mae={np.abs(fake.mean(0) - real_mean).mean():.4f} finite={np.isfinite(fake).all()}", flush=True)
