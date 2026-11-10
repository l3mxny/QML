"""Diagnostics of one decoder-comparison run. Usage: python collect_compare.py <row index in runs_small_circuits.csv>

Same metrics and same real-image references as collect_diagnostics.py, plus the exported circuit on Aer and fake_torino.
Writes plotdata/cmp_<row>.json and plotdata/cmp_<row>.npz.
"""
import contextlib
import io
import json
import os
import pickle
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plotdata")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(os.path.join(REPO, "apps", "logistics"))

import jax  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

from qugen.main.discriminator import discriminator_factory  # noqa: E402
from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler  # noqa: E402
from qugen.main.hardware import ibm, qiskit_export  # noqa: E402

jax.config.update("jax_enable_x64", True)
warnings.simplefilter("ignore", ConvergenceWarning)

row_index = int(sys.argv[1])
row = pd.read_csv(os.path.join(REPO, "experiments_configs", "runs_small_circuits.csv"), dtype=str, keep_default_na=False).iloc[row_index]
MODEL = os.path.basename(row.result_path)
assert MODEL.startswith("continuous"), row.result_path
DATA_SET = "mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area"
EPOCHS = [0, 100, 500] + list(range(1000, 10001, 1000))
FILM_EPOCHS = [0, 100, 500, 1000, 5000, 10000]
N = 10000
N_HARDWARE = 100
SHOTS = 4096

data = np.load(f"./training_data/{DATA_SET}.npy") / 255.
labels = np.load(f"./training_data/{DATA_SET}_labels.npy").astype(int)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0, stratify=labels)
real_shares = np.bincount(labels, minlength=10) / len(labels)
with open(os.path.join(OUT, "refs.pkl"), "rb") as f:
    clf_real, refs = pickle.load(f)


def fit_digit(x, y):
    return MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True, max_iter=200, random_state=0).fit(x, y)


def proba_10(clf, x):
    p = np.zeros((len(x), 10))
    p[:, clf.classes_] = clf.predict_proba(x)
    return p


def digit_scores(clf):
    p = proba_10(clf, x_test)
    per_digit = [roc_auc_score(y_test == d, p[:, d]) for d in range(10)]
    return float(accuracy_score(y_test, p.argmax(1))), float(np.mean(per_digit)), per_digit


def fit_on_generated(x, y):
    keep = np.bincount(y, minlength=10)[y] >= 10
    return fit_digit(x[keep], y[keep]) if len(np.unique(y[keep])) >= 2 else None


def c2st(a, b, sigma=0.05, seed=0):
    x = np.vstack([a, b])
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    x = x + np.random.default_rng(seed).normal(0, sigma, x.shape)
    x_a, x_b, y_a, y_b = train_test_split(x, y, test_size=0.3, random_state=seed, stratify=y)
    p = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(x_a, y_a).predict_proba(x_b)[:, 1]
    return float(accuracy_score(y_b, p > 0.5)), float(roc_auc_score(y_b, p))


def nn_rmse(query, pool):
    d, _ = NearestNeighbors(n_neighbors=1, n_jobs=2).fit(pool).kneighbors(query)
    return float(d.mean() / np.sqrt(query.shape[1]))


def pixel_metrics(fake, real):
    cov_f, cov_r = np.cov(fake.T), np.cov(real.T)
    return dict(mean_image_mae=float(np.abs(fake.mean(0) - real.mean(0)).mean()),
                std_image_mae=float(np.abs(fake.std(0) - real.std(0)).mean()),
                marginal_w1=float(np.mean([wasserstein_distance(fake[:, p], real[:, p]) for p in range(16)])),
                cov_rel_error=float(np.linalg.norm(cov_f - cov_r) / np.linalg.norm(cov_r)))


def coverage(samples):
    proba = proba_10(clf_real, samples)
    pseudo = proba.argmax(1)
    shares = np.bincount(pseudo, minlength=10) / len(samples)
    return pseudo, dict(class_shares=shares.tolist(), n_classes_covered=int((shares >= 0.5 * real_shares).sum()),
                        class_tv=float(0.5 * np.abs(shares - real_shares).sum()), confidence=float(proba.max(1).mean()))


model = ContinuousQGANModelHandler()
with contextlib.redirect_stdout(io.StringIO()):
    model.reload(model_name=MODEL, epoch=10000)
assert model.circuit_depth == [2, 1] and model.decoding_scheme == row.decoding_scheme, (model.circuit_depth, model.decoding_scheme)
D = discriminator_factory(model.discriminator_name)(is_critic=True, n_channels=1)


def critic_auc(weights, real, fake):
    score = lambda x: np.concatenate([np.asarray(D.apply(weights, x[i:i + 4096])).ravel() for i in range(0, len(x), 4096)])  # noqa: E731
    return float(roc_auc_score(np.r_[np.ones(len(real)), np.zeros(len(fake))], np.r_[score(real), score(fake)]))


rows, arrays = [], {}
for epoch in EPOCHS:
    with open(f"experiments/{MODEL}/parameters_training_iteration={epoch}.pickle", "rb") as f:
        model.generator_weights, critic_weights = pickle.load(f)
    model.random_key = jax.random.PRNGKey(2026)
    fake = np.asarray(model.sample(N)).reshape(N, -1)
    assert np.isfinite(fake).all(), f"non-finite samples at {epoch}"
    pseudo, result = coverage(fake)
    r = dict(epoch=epoch, critic_auc=critic_auc(critic_weights, x_test[:N], fake), **result)
    r["c2st_blur_acc"], r["c2st_blur_auc"] = c2st(x_test[:N], fake)
    clf_fake = fit_on_generated(fake, pseudo)
    r["tstr_acc"], r["tstr_auc"] = (float("nan"), float("nan")) if clf_fake is None else digit_scores(clf_fake)[:2]
    r.update(pixel_metrics(fake, x_test))
    r["fake_to_real_rmse"] = nn_rmse(fake[:2000], x_train[:N])
    r["real_to_fake_rmse"] = nn_rmse(x_test[:2000], fake)
    if epoch in FILM_EPOCHS:
        arrays[f"film_{epoch}"] = fake[:12]
    rows.append(r)
    print(MODEL[-8:], row.decoding_scheme, row.random_seed, epoch, {m: round(r[m], 3) for m in ["n_classes_covered", "class_tv", "c2st_blur_auc", "tstr_acc", "tstr_auc", "mean_image_mae"]}, flush=True)

grid = np.concatenate([np.where(pseudo == d)[0][:10] for d in range(10)])
arrays["grid"], arrays["grid_labels"] = fake[grid], pseudo[grid]
arrays["mean_final"], arrays["std_final"] = fake.mean(0), fake.std(0)

# Exported circuit: ideal device (Aer) and simulated device noise (fake_torino), final checkpoint
noise = qiskit_export.sample_noise(model, n_samples=N_HARDWARE, seed=3)
circuits = quiet = None
with contextlib.redirect_stdout(io.StringIO()):
    circuits = qiskit_export.generator_circuits(model, noise)
ideal = qiskit_export.ideal_probs(model, noise)
ideal = ideal / ideal.sum(1, keepdims=True)
images = dict(ideal=qiskit_export.decode_probs(model, ideal))
hardware = dict(logical=ibm.resource_summary(circuits[0]))
for backend_name in ["aer", "fake_torino"]:
    backend = ibm.get_backend(backend_name)
    isa, layout = ibm.transpile(circuits, backend)
    counts, _ = ibm.run_sampler(isa, backend, shots=SHOTS)
    probs = np.stack([qiskit_export.counts_to_probs(c, model.n_qubits, smoothing=0.5) for c in counts])
    images[backend_name] = qiskit_export.decode_probs(model, probs)
    _, result = coverage(images[backend_name])
    two_q = [ibm.two_qubit_gate_count(c) for c in isa]
    hardware[backend_name] = dict(fidelity=float(qiskit_export.classical_fidelity(probs, ideal).mean()),
                                  pixel_mae=float(np.abs(images[backend_name] - images["ideal"]).mean()),
                                  mean_image_mae=float(np.abs(images[backend_name].mean(0) - x_test.mean(0)).mean()),
                                  n_two_qubit_gates_mean=float(np.mean(two_q)), layout=[int(q) for q in layout],
                                  **{k: v for k, v in ibm.resource_summary(isa[0]).items() if k != "gate_counts"}, **result)
    print(MODEL[-8:], backend_name, {k: round(hardware[backend_name][k], 3) for k in ["fidelity", "pixel_mae", "n_two_qubit_gates_mean"]}, flush=True)
_, result = coverage(images["ideal"])
hardware["ideal"] = result
arrays.update({f"hw_{k}": v[:10] for k, v in images.items()})

np.savez(os.path.join(OUT, f"cmp_{row_index}.npz"), **arrays)
with open(os.path.join(OUT, f"cmp_{row_index}.json"), "w") as f:
    json.dump(dict(model=MODEL, decoding=row.decoding_scheme, seed=int(row.random_seed), rows=rows, hardware=hardware), f, indent=1, default=str)
print("done", flush=True)
