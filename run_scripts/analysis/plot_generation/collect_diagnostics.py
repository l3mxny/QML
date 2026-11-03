"""Per-checkpoint diagnostics of one baseline run. Usage: python collect_diagnostics.py <0|1>

Writes plotdata/diag_<k>.json (scalar rows) and plotdata/diag_<k>.npz (arrays).
All checkpoints sample from the same noise inputs, so differences between checkpoints are not sampling noise.
"""
import contextlib
import io
import json
import os
import pickle
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plotdata")
os.makedirs(OUT, exist_ok=True)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
os.chdir(os.path.join(REPO, "apps", "logistics"))

import jax  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.neighbors import NearestNeighbors  # noqa: E402
from sklearn.neural_network import MLPClassifier  # noqa: E402

from qugen.main.discriminator import discriminator_factory  # noqa: E402
from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler  # noqa: E402

jax.config.update("jax_enable_x64", True)  # as in train_image_qgan.main
warnings.simplefilter("ignore", ConvergenceWarning)

k = int(sys.argv[1])
MODEL = ["continuous_mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area_minmax_qgan_a33dbb10",
         "continuous_mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area_minmax_qgan_6538ebe1"][k]
DATA_SET = "mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area"
EPOCHS = [0, 100, 200, 300, 400] + list(range(500, 10001, 500))
ROC_EPOCHS = [100, 1000, 5000, 10000]
FILM_EPOCHS = [0, 100, 200, 500, 1000, 2000, 5000, 10000]
N = 10000

data = np.load(f"./training_data/{DATA_SET}.npy") / 255.
labels = np.load(f"./training_data/{DATA_SET}_labels.npy").astype(int)
# same split as digit_coverage.fit_classifier; every real image was also seen by the GAN during training
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0, stratify=labels)
real_shares = np.bincount(labels, minlength=10) / len(labels)


def fit_digit(x, y):
    return MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True, max_iter=200, random_state=0).fit(x, y)


def proba_10(clf, x):
    p = np.zeros((len(x), 10))
    p[:, clf.classes_] = clf.predict_proba(x)
    return p


def fit_on_generated(x, y):
    """Digit classifier on generated images, or None if they hold fewer than two usable digits (early iterations)."""
    counts = np.bincount(y, minlength=10)
    keep = counts[y] >= 10  # early stopping splits off a stratified validation set
    return fit_digit(x[keep], y[keep]) if len(np.unique(y[keep])) >= 2 else None


def digit_scores(clf):
    p = proba_10(clf, x_test)
    per_digit = [roc_auc_score(y_test == d, p[:, d]) for d in range(10)]
    return float(accuracy_score(y_test, p.argmax(1))), float(np.mean(per_digit)), per_digit, p


def c2st(a, b, sigma=0., seed=0):
    """Classifier two-sample test: held-out accuracy and AUC of telling a from b (0.5 = indistinguishable)."""
    x = np.vstack([a, b])
    y = np.r_[np.ones(len(a)), np.zeros(len(b))]
    if sigma > 0:  # blur the exact pixel values so the test judges image structure, not numerical precision
        x = x + np.random.default_rng(seed).normal(0, sigma, x.shape)
    x_a, x_b, y_a, y_b = train_test_split(x, y, test_size=0.3, random_state=seed, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(x_a, y_a)
    p = clf.predict_proba(x_b)[:, 1]
    return float(accuracy_score(y_b, p > 0.5)), float(roc_auc_score(y_b, p)), y_b, p


def nn_rmse(query, pool):
    d, _ = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(pool).kneighbors(query)
    return float(d.mean() / np.sqrt(query.shape[1]))


def pixel_metrics(fake, real):
    cov_f, cov_r = np.cov(fake.T), np.cov(real.T)
    return dict(mean_image_mae=float(np.abs(fake.mean(0) - real.mean(0)).mean()),
                std_image_mae=float(np.abs(fake.std(0) - real.std(0)).mean()),
                marginal_w1=float(np.mean([wasserstein_distance(fake[:, p], real[:, p]) for p in range(16)])),
                cov_rel_error=float(np.linalg.norm(cov_f - cov_r) / np.linalg.norm(cov_r)))


# Reference values from real data only, computed by run 0 and reused by run 1
REFS = os.path.join(OUT, "refs.pkl")
if k == 0 and not os.path.exists(REFS):
    clf_real = fit_digit(x_train, y_train)
    acc_full, auc_full, auc_digit_full, p_real = digit_scores(clf_real)
    acc_10k, auc_10k, _, _ = digit_scores(fit_digit(x_train[:N], y_train[:N]))
    refs = dict(digit_accuracy_48k=acc_full, digit_auc_48k=auc_full, digit_accuracy_10k=acc_10k, digit_auc_10k=auc_10k,
                digit_auc_per_digit=auc_digit_full, heldout_confidence=float(p_real.max(1).mean()),
                digit_recall=[float((p_real.argmax(1)[y_test == d] == d).mean()) for d in range(10)])
    refs["c2st_acc"], refs["c2st_auc"], *_ = c2st(x_test[:N], x_train[:N])
    refs["c2st_blur_acc"], refs["c2st_blur_auc"], *_ = c2st(x_test[:N], x_train[:N], sigma=0.05)
    refs.update({f"real_{m}": v for m, v in pixel_metrics(x_test[:N], x_train[:N]).items()})
    refs["real_to_real_rmse"] = nn_rmse(x_test[:2000], x_train[:N])
    with open(REFS + ".tmp", "wb") as f:
        pickle.dump((clf_real, refs), f)
    os.replace(REFS + ".tmp", REFS)
while not os.path.exists(REFS):
    time.sleep(5)
with open(REFS, "rb") as f:
    clf_real, refs = pickle.load(f)
print("references", {m: round(v, 4) for m, v in refs.items() if isinstance(v, float)}, flush=True)

model = ContinuousQGANModelHandler()
with contextlib.redirect_stdout(io.StringIO()):
    model.reload(model_name=MODEL, epoch=10000)
assert np.allclose(np.asarray(model.reverse_lookup, dtype=float), [0., 1.]), model.reverse_lookup  # same pixel scale
D = discriminator_factory(model.discriminator_name)(is_critic=True, n_channels=1)


def critic_scores(weights, x):
    return np.concatenate([np.asarray(D.apply(weights, x[i:i + 4096])).ravel() for i in range(0, len(x), 4096)])


rows, arrays = [], dict(mean_real=x_test.mean(0), std_real=x_test.std(0))
for epoch in EPOCHS:
    with open(f"experiments/{MODEL}/parameters_training_iteration={epoch}.pickle", "rb") as f:
        model.generator_weights, critic_weights = pickle.load(f)
    model.random_key = jax.random.PRNGKey(2026)
    fake = np.asarray(model.sample(N)).reshape(N, -1)
    row = dict(epoch=epoch, nan_share=float(np.isnan(fake).any(1).mean()))
    assert row["nan_share"] == 0, f"NaN samples at {epoch}"

    # 1. Can the training critic tell held-out-from-the-classifier real images from generated ones?
    real = x_test[:N]
    s_real, s_fake = critic_scores(critic_weights, real), critic_scores(critic_weights, fake)
    y = np.r_[np.ones(N), np.zeros(N)]
    s = np.r_[s_real, s_fake]
    row["critic_auc"] = float(roc_auc_score(y, s))
    half = np.random.default_rng(0).permutation(2 * N)
    a, b = half[:N], half[N:]
    candidates = np.quantile(s[a], np.linspace(0.01, 0.99, 197))
    threshold = candidates[np.argmax([accuracy_score(y[a], s[a] > t) for t in candidates])]
    row["critic_acc"] = float(accuracy_score(y[b], s[b] > threshold))

    # 2. Can an independent classifier tell them apart?
    row["c2st_acc"], row["c2st_auc"], y_b, p = c2st(real, fake)
    row["c2st_blur_acc"], row["c2st_blur_auc"], y_b_blur, p_blur = c2st(real, fake, sigma=0.05)
    if epoch in ROC_EPOCHS:
        fpr, tpr, _ = roc_curve(y_b_blur, p_blur)
        keep = np.unique(np.linspace(0, len(fpr) - 1, 300).astype(int))
        arrays[f"roc_blur_fpr_{epoch}"], arrays[f"roc_blur_tpr_{epoch}"] = fpr[keep], tpr[keep]
        fpr, tpr, _ = roc_curve(y[b], s[b])
        keep = np.unique(np.linspace(0, len(fpr) - 1, 300).astype(int))
        arrays[f"roc_critic_fpr_{epoch}"], arrays[f"roc_critic_tpr_{epoch}"] = fpr[keep], tpr[keep]

    # 3. Digit structure: real-data classifier on generated images, and a classifier trained on generated images
    proba = proba_10(clf_real, fake)
    pseudo = proba.argmax(1)
    shares = np.bincount(pseudo, minlength=10) / N
    row.update(class_shares=shares.tolist(), n_classes_covered=int((shares >= 0.5 * real_shares).sum()),
               class_tv=float(0.5 * np.abs(shares - real_shares).sum()), confidence=float(proba.max(1).mean()))
    clf_fake = fit_on_generated(fake, pseudo)
    row["tstr_acc"], row["tstr_auc"], tstr_per_digit = (float("nan"), float("nan"), None) if clf_fake is None \
        else digit_scores(clf_fake)[:3]

    # 4. Pixel errors against real images
    row.update(pixel_metrics(fake, x_test))
    row["fake_to_real_rmse"] = nn_rmse(fake[:2000], x_train[:N])
    row["real_to_fake_rmse"] = nn_rmse(x_test[:2000], fake)
    arrays[f"mean_{epoch}"], arrays[f"std_{epoch}"] = fake.mean(0), fake.std(0)
    if epoch in FILM_EPOCHS:
        arrays[f"film_{epoch}"] = fake[:12]
    if epoch == EPOCHS[-1]:
        row["tstr_auc_per_digit"] = tstr_per_digit
        for d in range(10):
            arrays[f"digit_mean_real_{d}"] = x_test[y_test == d].mean(0)
            arrays[f"digit_mean_fake_{d}"] = fake[pseudo == d].mean(0)
        real_idx = np.concatenate([np.where(y_test == d)[0][:10] for d in range(10)])
        arrays["real_grid"], arrays["real_grid_labels"] = x_test[real_idx], y_test[real_idx]
        grid = np.concatenate([np.where(pseudo == d)[0][:10] for d in range(10)])
        arrays["grid"], arrays["grid_labels"] = fake[grid], pseudo[grid]
    rows.append(row)
    print(epoch, {m: round(row[m], 3) for m in ["critic_auc", "c2st_auc", "c2st_blur_auc", "tstr_acc", "tstr_auc",
                                                "n_classes_covered", "class_tv", "mean_image_mae",
                                                "fake_to_real_rmse"]}, flush=True)
    with open(os.path.join(OUT, f"diag_{k}.json"), "w") as f:  # keep partial progress
        json.dump(dict(model=MODEL, noise_tuning=model.noise_tuning, refs=refs, real_shares=real_shares.tolist(),
                       rows=rows), f, indent=1)

np.savez(os.path.join(OUT, f"diag_{k}.npz"), **arrays)
print("done", flush=True)
