"""Render the diagnostic figures from plotdata/ into the repo's plots_small_circuits/ folder."""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "plotdata")
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
OUT = os.path.join(REPO, "plots_small_circuits")
EXPERIMENTS = os.path.join(REPO, "apps", "logistics", "experiments")
BASELINES = {"K = 10": "continuous_mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area_minmax_qgan_a33dbb10",
             "K = 20": "continuous_mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area_minmax_qgan_6538ebe1"}

# Reference palette (light mode), validated with the skill's script
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
COLORS = dict(zip(BASELINES, [BLUE, ORANGE]))
DIVERGING = LinearSegmentedColormap.from_list("div", ["#d03b3b", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "axes.titlesize": 10,
    "axes.titleweight": "semibold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "legend.frameon": False, "lines.linewidth": 1.6,
})


def save(fig, name, note=None):
    if note:  # below the figure; bbox_inches="tight" grows the canvas to include it
        fig.text(0.01, -0.01, note, color=MUTED, fontsize=7.5, ha="left", va="top", wrap=True)
    fig.savefig(os.path.join(OUT, name), dpi=170, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name, flush=True)


def diag():
    out = {}
    for k, label in enumerate(BASELINES):
        with open(os.path.join(DATA, f"diag_{k}.json")) as f:
            d = json.load(f)
        d["df"] = pd.DataFrame(d["rows"])
        d["arrays"] = np.load(os.path.join(DATA, f"diag_{k}.npz"))
        out[label] = d
    return out


def show_image(ax, img, vmax=1.):
    ax.imshow(np.asarray(img).reshape(4, 4), cmap="gray", vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def curves(ax, data, column, title, reference=None, reference_label=None, ylabel=None, marker=True, skip_first=0):
    for label, d in data.items():
        df = d["df"].iloc[skip_first:]
        ax.plot(df["epoch"], df[column], color=COLORS[label], label=label, marker="o" if marker else None, ms=3.5)
    if reference is not None:
        ax.axhline(reference, color=MUTED, lw=0.9)
        ax.text(1.0, reference, f" {reference_label}", transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=7.5, color=INK_2)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("Training iteration")
    if ylabel:
        ax.set_ylabel(ylabel)


# ------------------------------------------------------------------ existing figures, refreshed on final checkpoints
def fig_trained_export():
    d = np.load(os.path.join(DATA, "B.npz"))
    info = json.load(open(os.path.join(DATA, "B.json")))["K10"]
    rows = [("ideal", "PennyLane (training)"), ("before", "Qiskit export, before fix"),
            ("after", "Qiskit export, after fix"), ("aer", "Aer, 4096 shots"),
            ("fake_torino", "fake_torino (device noise),\n4096 shots")]
    n = 10
    fig, axs = plt.subplots(len(rows), n + 1, figsize=(1.0 * n + 2.6, 0.92 * len(rows) + 0.3),
                            gridspec_kw=dict(width_ratios=[2.6] + [1] * n, wspace=0.08, hspace=0.12))
    fig.subplots_adjust(top=0.9)
    for r, (key, label) in enumerate(rows):
        fid, err = d[f"K10_fid_{key}"].mean(), np.abs(d[f"K10_img_{key}"] - d["K10_img_ideal"]).mean()
        axs[r, 0].axis("off")
        axs[r, 0].text(1.0, 0.62, label, ha="right", va="center", fontsize=8.5, transform=axs[r, 0].transAxes)
        if key != "ideal":
            axs[r, 0].text(1.0, 0.16, f"fidelity {fid:.3f}, pixel MAE {err:.3f}", ha="right", va="center", fontsize=7.5,
                           color=INK_2, transform=axs[r, 0].transAxes)
        for c in range(n):
            show_image(axs[r, c + 1], d[f"K10_img_{key}"][c])
    t = info["fake_torino"]
    fig.suptitle("Trained 4x4 baseline (K = 10, iteration 10,000) exported and sampled", x=0.01, ha="left",
                 fontweight="semibold", fontsize=11, y=1.0)
    save(fig, "02_trained_model_export.png",
         f"First 10 of 20 noise inputs shown; fidelity (classical, to PennyLane) and pixel MAE are over all 20. Circuit: "
         f"{info['logical']['n_two_qubit_gates']} two-qubit gates, {t['n_two_qubit_gates']} CZ after routing on "
         f"fake_torino (two-qubit depth {t['two_qubit_depth']}, depth {t['depth']}).")


def fig_shot_noise():
    d = np.load(os.path.join(DATA, "C.npz"))
    shots = [128, 512, 2048, 8192, 0]
    n = 6
    fig, axs = plt.subplots(len(shots), n + 1, figsize=(1.0 * n + 1.6, 0.92 * len(shots) + 0.3),
                            gridspec_kw=dict(width_ratios=[1.6] + [1] * n, wspace=0.08, hspace=0.12))
    fig.subplots_adjust(top=0.9)
    for r, s in enumerate(shots):
        axs[r, 0].axis("off")
        axs[r, 0].text(1.0, 0.5, "exact" if s == 0 else f"{s} shots", ha="right", va="center", fontsize=9,
                       transform=axs[r, 0].transAxes)
        for c in range(n):
            show_image(axs[r, c + 1], d[f"K10_shots_{s}"][c])
    fig.suptitle("Shot noise on decoded 4x4 FRQI images (K = 10, iteration 10,000)", x=0.01, ha="left",
                 fontweight="semibold", fontsize=11, y=1.0)
    save(fig, "03_shot_noise_images.png", "True multinomial sampling (NumPy) of the exact generator probabilities. Figure 16 has the error.")


def fig_training_checks():
    d, info = np.load(os.path.join(DATA, "D.npz")), json.load(open(os.path.join(DATA, "D.json")))
    d2, info2 = np.load(os.path.join(DATA, "D2.npz")), json.load(open(os.path.join(DATA, "D2.json")))
    # exact, shots (cache on), shots (cache off), amplitude
    panels = [(d, 0, info[0]), (d, 1, info[1]), (d, 2, info[2]), (d, 3, info[3]), (d2, 0, info2[0]), (d2, 1, info2[1])]
    fig, axs = plt.subplots(2, 3, figsize=(10.5, 5.2))
    axs = axs.ravel()
    for ax, (src, i, run) in zip(axs, panels):
        it, gen, disc = src[f"{i}__iteration"], src[f"{i}__loss_generator"], src[f"{i}__loss_discriminator"]
        ax.plot(it, disc, color=ORANGE, label="Critic loss")
        ax.plot(it, gen, color=BLUE, label="Generator loss")
        ax.set_xlim(0, it.max())
        ax.set_xlabel("Iteration")
        ax.set_title(f"{run['name']}  ({run['seconds']:.0f} s)", fontsize=9)
        if run["first_nan"] is None:
            ax.text(0.97, 0.55, "✓ finite for all 150 iterations", transform=ax.transAxes, ha="right", fontsize=8, color="#006300")
        else:
            ax.axvline(run["first_nan"], color=CRITICAL, lw=1)
            ax.text(0.97, 0.05, f"✗ NaN from iteration {run['first_nan']}", transform=ax.transAxes, ha="right",
                    fontsize=8, color=CRITICAL)
    for ax in axs[[0, 3]]:
        ax.set_ylabel("Loss (WGAN-GP)")
    axs[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Short training runs: shots failed with the on-disk compilation cache, and run clean after removing it", x=0.01, ha="left",
                 fontweight="semibold", fontsize=11)
    fig.tight_layout()
    save(fig, "04_training_checks.png",
         "4x4 area MNIST, 8192 images, batch 64, 10 critic steps per iteration, x64 set after importing the handler (as train_image_qgan.py). "
         "FRQI: circuit c01, depth (2,1), 5 qubits. Amplitude: circuit c05, depth 4, 4 qubits. NaN checks off so a failing run keeps logging. "
         "Top row and bottom left: before the fix. Bottom middle and right: after removing the cache from the handler.")


def fig_compiled_step():
    d, info = np.load(os.path.join(DATA, "E.npz")), json.load(open(os.path.join(DATA, "E.json")))
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.0))
    labels = ["Before\n(uncompiled)", "Loss + gradient\ncompiled", "+ optimizer\ncompiled"]
    ms = [1848, 866, 458]  # steady state measured on 2026-10-20: (8,2) circuit, 10 critic steps, batch 64, x64 + debug_nans
    bars = axs[0].bar(labels, ms, width=0.55, color=BLUE)
    for b, v in zip(bars, ms):
        axs[0].text(b.get_x() + b.get_width() / 2, v + 30, f"{v} ms" + (f"\n{ms[0] / v:.1f}x faster" if v != ms[0] else ""),
                    ha="center", va="bottom", fontsize=8)
    axs[0].set_ylim(0, ms[0] * 1.25)
    axs[0].set_ylabel("Time per training iteration")
    axs[0].grid(axis="x", visible=False)
    axs[0].set_title("Training step time, depth (8,2)")
    a, b = d["0__loss_discriminator"], d["1__loss_discriminator"]
    axs[1].plot(d["0__iteration"], a, color=AXIS, lw=4, label=info[0]["name"])
    axs[1].plot(d["1__iteration"], b, color=BLUE, lw=0, marker="o", ms=4, label=info[1]["name"])
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Critic loss")
    axs[1].legend(loc="center right", fontsize=8)
    axs[1].set_title("Same seed, uncompiled vs compiled, depth (2,1)")
    fig.suptitle("Compiling the training step: same computation, about 4x faster", x=0.01, ha="left", fontweight="semibold",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "05_compiled_training_step.png",
         f"Step times measured on this machine (14-core CPU, jax 0.6.0) with x64 and jax_debug_nans on, excluding compilation. The runs are not "
         f"bit-identical: critic losses differ by {np.abs(a - b)[:9].max():.0e} over the first 9 iterations (rounding), then drift to at most {np.abs(a - b).max():.3f} by iteration 40.")


def fig_progress(data):
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.2))
    curves(axs[0], data, "n_classes_covered", "Digits generated (of 10)")
    curves(axs[1], data, "class_tv", "Class distribution distance to real data")
    curves(axs[2], data, "confidence", "Digit classifier confidence on generated images",
           reference=next(iter(data.values()))["refs"]["heldout_confidence"], reference_label="real")
    axs[0].axhline(10, color=AXIS, lw=0.8)
    axs[0].set_ylim(0, 10.8), axs[0].set_yticks(range(0, 11, 2))
    axs[1].set_ylim(bottom=0)
    axs[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Digit coverage over training, depth (8,2), 5 qubits", x=0.01, ha="left", fontweight="semibold", fontsize=11)
    fig.tight_layout()
    save(fig, "06_baseline_progress.png",
         "10,000 samples per checkpoint from the same noise inputs. A digit counts as generated if its share is at least "
         "half its real share. Distance = total variation (0 identical, 1 disjoint).")

    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    x, w = np.arange(10), 0.26
    real = np.asarray(next(iter(data.values()))["real_shares"])
    ax.bar(x - w, real, w - 0.02, color=AXIS, label="Real data")
    top = real.max()
    for k, (label, d) in enumerate(data.items()):
        shares = d["df"]["class_shares"].iloc[-1]
        ax.bar(x + k * w, shares, w - 0.02, color=COLORS[label], label=f"{label}")
        top = max(top, max(shares))
    ax.plot(np.c_[x - 0.45, x + 0.45].T, np.c_[0.5 * real, 0.5 * real].T, color=INK_2, lw=1)
    ax.plot([], [], color=INK_2, lw=1, label="Coverage threshold (half the real share)")
    ax.set_xticks(x, [str(i) for i in x])
    ax.set_xlabel("Digit (classifier prediction)")
    ax.set_ylabel("Share of samples")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, ncols=2, loc="upper left")
    ax.set_ylim(0, top * 1.45)
    ax.set_title("Which digits the baselines generate, iteration 10,000", fontsize=11)
    save(fig, "07_baseline_class_shares.png", "10,000 samples per model, classified by the real-data digit classifier.")


def fig_samples(data):
    first = next(iter(data.values()))["arrays"]
    grids = [("Real MNIST 4x4 (area)", first["real_grid"], first["real_grid_labels"])]
    for label, d in data.items():
        grids.append((f"Generated, {label}, iteration 10,000", d["arrays"]["grid"], d["arrays"]["grid_labels"]))
    fig = plt.figure(figsize=(13, 5.0))
    outer = fig.add_gridspec(1, 3, wspace=0.12)
    for g, (title, imgs, labels) in enumerate(grids):
        inner = outer[g].subgridspec(10, 11, wspace=0.06, hspace=0.06, width_ratios=[0.6] + [1] * 10)
        for digit in range(10):
            ax = fig.add_subplot(inner[digit, 0])
            ax.axis("off")
            ax.text(0.5, 0.5, str(digit), ha="center", va="center", fontsize=8, color=INK_2)
            row = imgs[labels == digit][:10]
            for c in range(10):
                ax = fig.add_subplot(inner[digit, c + 1])
                show_image(ax, row[c]) if c < len(row) else ax.axis("off")
        fig.text(outer[g].get_position(fig).x0, 0.93, title, fontsize=10, fontweight="semibold", ha="left")
    save(fig, "08_baseline_samples.png", "Rows = digit: true label for real images, classifier prediction for generated images.")


# ------------------------------------------------------------------ new diagnostics
def fig_loss(data):
    fig, axs = plt.subplots(2, 2, figsize=(12, 6), sharex=True)
    panels = [("loss_discriminator", "Critic loss", axs[0, 0], False), ("loss_generator", "Generator loss", axs[0, 1], False),
              ("grad_mag_discriminator", "Critic gradient norm", axs[1, 0], True),
              ("grad_mag_generator", "Generator gradient norm", axs[1, 1], True)]
    logs = {label: pd.read_csv(os.path.join(EXPERIMENTS, name, "evaluation_summary.csv")) for label, name in BASELINES.items()}
    for column, title, ax, log_scale in panels:
        low, high = [], []
        for label, log in logs.items():
            ax.plot(log["iteration"], log[column], color=COLORS[label], lw=0.3, alpha=0.22)
            smooth = log[column].rolling(200, min_periods=20, center=True).mean()
            ax.plot(log["iteration"], smooth, color=COLORS[label], lw=1.8, label=label)
            low.append(smooth.iloc[300:].min()), high.append(smooth.iloc[300:].max())
        if log_scale:
            ax.set_yscale("log")
        else:
            lo, hi = min(low), max(high)
            ax.set_ylim(lo - 0.6 * (hi - lo), hi + 0.6 * (hi - lo))  # the first iterations are far outside this range
        ax.set_title(title)
    for ax in axs[1]:
        ax.set_xlabel("Training iteration")
    axs[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Training losses and gradient norms, depth (8,2), 5 qubits", x=0.01, ha="left", fontweight="semibold", fontsize=11)
    fig.tight_layout()
    save(fig, "09_loss_and_gradient_curves.png",
         "Faint: every iteration. Solid: 200-iteration rolling mean. Critic loss includes the gradient penalty; the y-axis of "
         "the loss panels is zoomed past the first ~300 iterations, where the critic loss starts near 8.")


def roc_panel(ax, data, label, kind, title):
    arrays = data[label]["arrays"]
    ramp = {100: "#86b6ef", 1000: "#5598e7", 5000: "#256abf", 10000: "#0d366b"}
    ax.plot([0, 1], [0, 1], color=AXIS, lw=0.9)
    for epoch, color in ramp.items():
        ax.plot(arrays[f"roc_{kind}_fpr_{epoch}"], arrays[f"roc_{kind}_tpr_{epoch}"], color=color, label=f"it. {epoch:,}")
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")


def fig_separability(data):
    refs = next(iter(data.values()))["refs"]
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.3))
    curves(axs[0, 0], data, "critic_auc", "Training critic: AUC", reference=0.5, reference_label="chance")
    curves(axs[0, 1], data, "c2st_blur_auc", "Independent classifier: AUC", reference=refs["c2st_blur_auc"], reference_label="real vs real")
    curves(axs[0, 2], data, "c2st_blur_acc", "Independent classifier: accuracy", reference=refs["c2st_blur_acc"], reference_label="real vs real")
    curves(axs[1, 0], data, "critic_acc", "Training critic: accuracy", reference=0.5, reference_label="chance")
    for ax in axs[0]:
        ax.set_ylim(0.45, 1.02)
    axs[1, 0].set_ylim(0.45, 1.02)
    axs[0, 0].legend(loc="upper right", fontsize=8)
    label = "K = 10"
    roc_panel(axs[1, 1], data, label, "blur", f"ROC, independent classifier ({label})")
    roc_panel(axs[1, 2], data, label, "critic", f"ROC, training critic ({label})")
    axs[1, 1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Accuracy and AUC: can real and generated images be told apart? (0.5 = no, 1.0 = always)", x=0.01,
                 ha="left", fontweight="semibold", fontsize=11)
    fig.tight_layout()
    raw = min(d["df"]["c2st_auc"].iloc[1:].min() for d in data.values())
    arrays = next(iter(data.values()))["arrays"]
    zeros_real, zeros_fake = (arrays["real_grid"] == 0).mean(), (arrays["grid"] == 0).mean()
    save(fig, "10_real_vs_generated_accuracy_auc.png",
         "10,000 real vs 10,000 generated images. Independent classifier: gradient boosting, 30% held out, pixels blurred with Gaussian noise "
         f"(sigma 0.05) so it judges image structure. Unblurred, its AUC is at least {raw:.3f} at every checkpoint after iteration 0: {zeros_real:.0%} of "
         f"real pixels are exactly 0 and {zeros_fake:.0%} of generated pixels are, which separates them regardless of image quality. Critic accuracy: threshold chosen "
         "on one half, scored on the other. Real images were seen by the GAN in training (no MNIST test set locally).")


def fig_digit_task(data):
    refs = next(iter(data.values()))["refs"]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.3), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    curves(axs[0], data, "tstr_acc", "Accuracy on real digits", reference=refs["digit_accuracy_10k"], reference_label="trained on real")
    curves(axs[1], data, "tstr_auc", "Macro AUC on real digits", reference=refs["digit_auc_10k"], reference_label="trained on real")
    axs[0].set_ylim(0, 1), axs[1].set_ylim(0.5, 1.0)
    axs[0].legend(loc="lower right", fontsize=8)
    x = np.arange(10)
    axs[2].bar(x - 0.27, refs["digit_auc_per_digit"], 0.25, color=AXIS, label="Trained on real")
    for k, (label, d) in enumerate(data.items()):
        axs[2].bar(x + k * 0.27, d["df"]["tstr_auc_per_digit"].iloc[-1], 0.25, color=COLORS[label], label=label)
    axs[2].set_ylim(0.5, 1.09)
    axs[2].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axs[2].set_xticks(x, [str(i) for i in x])
    axs[2].grid(axis="x", visible=False)
    axs[2].set_xlabel("Digit")
    axs[2].set_title("One-vs-rest AUC per digit, iteration 10,000", fontsize=9.5)
    axs[2].legend(fontsize=7.5, ncols=3, loc="upper left")
    fig.suptitle("Accuracy and AUC: do generated images carry digit information? (classifier trained on generated, tested on real)",
                 x=0.01, ha="left", fontweight="semibold", fontsize=11)
    fig.tight_layout()
    save(fig, "11_digit_task_accuracy_auc.png",
         "Generators have no labels, so 10,000 generated images were labelled by a classifier trained on real images, then a "
         "fresh classifier was trained on them and tested on 12,000 real images with true labels. 'Trained on real' uses 10,000 "
         f"real images. With all 48,000 real training images: accuracy {refs['digit_accuracy_48k']:.3f}, macro AUC {refs['digit_auc_48k']:.3f}.")


def fig_pixel_curves(data):
    refs = next(iter(data.values()))["refs"]
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.2))
    panels = [("mean_image_mae", "Mean image error (MAE)", "real_mean_image_mae"),
              ("std_image_mae", "Pixel std-dev error (MAE)", "real_std_image_mae"),
              ("marginal_w1", "Pixel distribution distance (Wasserstein-1)", "real_marginal_w1"),
              ("cov_rel_error", "Pixel covariance error (relative)", "real_cov_rel_error"),
              ("fake_to_real_rmse", "Generated to nearest real image (RMSE)", "real_to_real_rmse"),
              ("real_to_fake_rmse", "Real to nearest generated image (RMSE)", "real_to_real_rmse")]
    for ax, (column, title, ref) in zip(axs.flat, panels):
        curves(ax, data, column, title, reference=refs[ref], reference_label="real vs real", skip_first=1)
        ax.set_ylim(bottom=0)
    axs[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Pixel errors against real images over training", x=0.01, ha="left", fontweight="semibold", fontsize=11)
    fig.tight_layout()
    save(fig, "12_pixel_error_curves.png",
         "Pixels in [0, 1], 16 pixels per image, from iteration 100 on. 'Real vs real' compares two disjoint sets of real images: "
         "the lowest error sampling alone allows.")


def fig_mmd():
    fig, axs = plt.subplots(1, 3, figsize=(11, 2.9))
    for ax, (column, title) in zip(axs, [("mmd_rbf_n2000", "RBF MMD"), ("mmd_linear_n2000", "Linear MMD"), ("mmd_poly_n2000", "Polynomial MMD")]):
        for label, name in BASELINES.items():
            log = pd.read_csv(os.path.join(EXPERIMENTS, name, "evaluation_summary.csv")).dropna(subset=[column])
            log = log[log["iteration"] >= 1000]
            ax.plot(log["iteration"], log[column], color=COLORS[label], marker="o", ms=3.5, label=label)
        ax.set_title(title)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Training iteration")
    axs[0].legend(fontsize=8)
    fig.suptitle("Maximum mean discrepancy to real images", x=0.01, ha="left", fontweight="semibold", fontsize=11)
    fig.tight_layout()
    save(fig, "13_mmd_curves.png", "2000 samples per checkpoint, every 1000 iterations. Iteration 0 (RBF about 1.5) is left out so the rest is readable.")


def fig_pixel_maps(data):
    n = len(data)
    fig, axs = plt.subplots(n, 5, figsize=(11, 2.25 * n))
    for r, (label, d) in enumerate(data.items()):
        a = d["arrays"]
        panels = [(a["mean_real"], "Real mean", "gray", 0, 1), (a["mean_10000"], "Generated mean", "gray", 0, 1),
                  (a["mean_10000"] - a["mean_real"], "Mean error (generated − real)", DIVERGING, -0.1, 0.1),
                  (a["std_10000"] - a["std_real"], "Std-dev error (generated − real)", DIVERGING, -0.1, 0.1)]
        axs[r, 0].axis("off")
        axs[r, 0].text(0.5, 0.5, label, ha="center", va="center", fontsize=11, fontweight="semibold", transform=axs[r, 0].transAxes)
        for c, (img, title, cmap, vmin, vmax) in enumerate(panels, start=1):
            ax = axs[r, c]
            im = ax.imshow(img.reshape(4, 4), cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=8.5)
            for (i, j), v in np.ndenumerate(img.reshape(4, 4)):
                dark = (cmap == "gray" and v < 0.45) or (cmap != "gray" and abs(v) > 0.07)
                ax.text(j, i, f"{v:.2f}".replace("0.", "."), ha="center", va="center", fontsize=6.5, color=SURFACE if dark else INK)
    fig.suptitle("Per-pixel error at iteration 10,000", x=0.01, ha="left", fontweight="semibold", fontsize=11)
    fig.tight_layout()
    save(fig, "14_pixel_error_heatmaps.png", "Blue = generated too bright, red = too dark, gray = no error. Scale ±0.1.")


def fig_per_digit(data):
    fig = plt.figure(figsize=(13, 5.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.14)
    inner = outer[0].subgridspec(5, 11, wspace=0.08, hspace=0.1, width_ratios=[1.5] + [1] * 10)
    rows = [("Real", "digit_mean_real", None)]
    for label, d in data.items():
        rows.append((f"{label}", "digit_mean_fake", d))
    first = next(iter(data.values()))["arrays"]
    for digit in range(10):
        ax = fig.add_subplot(inner[0, digit + 1])
        show_image(ax, first[f"digit_mean_real_{digit}"], vmax=0.8)
        ax.set_title(str(digit), fontsize=9, loc="center")
    for r, (label, d) in enumerate(data.items(), start=1):
        for digit in range(10):
            show_image(fig.add_subplot(inner[r, digit + 1]), d["arrays"][f"digit_mean_fake_{digit}"], vmax=0.8)
    for r, text in enumerate(["Real", *data]):
        ax = fig.add_subplot(inner[r, 0])
        ax.axis("off")
        ax.text(0.95, 0.5, f"{text}\nmean image" if r == 0 else f"{text}\nmean image", ha="right", va="center", fontsize=8.5, transform=ax.transAxes)
    for r, (label, d) in enumerate(data.items(), start=3):
        ax = fig.add_subplot(inner[r, 0])
        ax.axis("off")
        ax.text(0.95, 0.5, f"{label}\nerror", ha="right", va="center", fontsize=8.5, transform=ax.transAxes)
        for digit in range(10):
            ax = fig.add_subplot(inner[r, digit + 1])
            diff = d["arrays"][f"digit_mean_fake_{digit}"] - first[f"digit_mean_real_{digit}"]
            ax.imshow(diff.reshape(4, 4), cmap=DIVERGING, vmin=-0.25, vmax=0.25, interpolation="nearest")
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
    ax = fig.add_subplot(outer[1])
    x = np.arange(10)
    for k, (label, d) in enumerate(data.items()):
        mae = [np.abs(d["arrays"][f"digit_mean_fake_{i}"] - first[f"digit_mean_real_{i}"]).mean() for i in x]
        ax.bar(x + (k - 0.5) * 0.38, mae, 0.36, color=COLORS[label], label=label)
    ax.set_xticks(x, [str(i) for i in x])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Mean image error (MAE)")
    ax.set_title("Mean-image error per digit")
    ax.legend(fontsize=8)
    fig.suptitle("Per-digit mean images at iteration 10,000", x=0.1, ha="left", fontweight="semibold", fontsize=11, y=0.98)
    save(fig, "15_per_digit_pixel_error.png",
         "Generated images are grouped by the digit the classifier predicts, real images by true label. Error scale ±0.25: blue "
         "= generated too bright, red = too dark.")


def fig_hardware_error():
    rows = pd.DataFrame(json.load(open(os.path.join(DATA, "C.json"))))
    d = np.load(os.path.join(DATA, "B.npz"))
    info = json.load(open(os.path.join(DATA, "B.json")))
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
    for key, label in [("K10", "K = 10"), ("K20", "K = 20")]:
        df = rows[rows["model"] == key]
        axs[0].plot(df["shots"], df["pixel_mae"], color=COLORS[label], marker="o", ms=4, label=label)
    axs[0].set_xscale("log", base=2)
    axs[0].set_xticks([128, 256, 512, 1024, 2048, 4096, 8192], ["128", "256", "512", "1024", "2048", "4096", "8192"])
    axs[0].minorticks_off()
    axs[0].set_ylim(bottom=0)
    axs[0].set_xlabel("Shots per image")
    axs[0].set_ylabel("Pixel MAE against exact image")
    axs[0].set_title("Shot noise alone")
    axs[0].legend(fontsize=8)
    groups = [("aer", "Aer\n4096 shots"), ("fake_torino", "fake_torino\n4096 shots")]
    x = np.arange(len(groups))
    for k, (key, label) in enumerate([("K10", "K = 10"), ("K20", "K = 20")]):
        values = [np.abs(d[f"{key}_img_{g}"] - d[f"{key}_img_ideal"]).mean() for g, _ in groups]
        bars = axs[1].bar(x + (k - 0.5) * 0.38, values, 0.36, color=COLORS[label], label=label)
        for b, v in zip(bars, values):
            axs[1].text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    axs[1].set_xticks(x, [g[1] for g in groups])
    axs[1].grid(axis="x", visible=False)
    axs[1].set_title("Exported circuit")
    axs[1].set_ylabel("Pixel MAE against exact image")
    fig.suptitle("Pixel error from sampling and device noise, iteration 10,000", x=0.01, ha="left", fontweight="semibold", fontsize=11)
    fig.tight_layout()
    fids = {k: {b: float(d[f"{k}_fid_{b}"].mean()) for b in ["aer", "fake_torino"]} for k in ["K10", "K20"]}
    save(fig, "16_hardware_pixel_error.png",
         f"Left: true multinomial sampling (NumPy), 500 noise inputs, 3 seeds. Right: 20 noise inputs. Mean fidelity on fake_torino: K=10 {fids['K10']['fake_torino']:.3f}, "
         f"K=20 {fids['K20']['fake_torino']:.3f} ({info['K10']['fake_torino']['n_two_qubit_gates']} and "
         f"{info['K20']['fake_torino']['n_two_qubit_gates']} CZ gates).")


def fig_filmstrip(data):
    epochs = [0, 100, 200, 500, 1000, 2000, 5000, 10000]
    n = len(data)
    fig = plt.figure(figsize=(1.0 * 10 + 1.6, 0.95 * len(epochs) * n * 0.52 + 0.8))
    outer = fig.add_gridspec(1, n, wspace=0.08)
    for g, (label, d) in enumerate(data.items()):
        inner = outer[g].subgridspec(len(epochs), 7, wspace=0.08, hspace=0.1, width_ratios=[1.4] + [1] * 6)
        for r, epoch in enumerate(epochs):
            ax = fig.add_subplot(inner[r, 0])
            ax.axis("off")
            ax.text(0.95, 0.5, f"{epoch:,}", ha="right", va="center", fontsize=8.5, transform=ax.transAxes)
            for c in range(6):
                show_image(fig.add_subplot(inner[r, c + 1]), d["arrays"][f"film_{epoch}"][c])
        fig.text(outer[g].get_position(fig).x0 + 0.05, 0.91, label, fontsize=10, fontweight="semibold")
    fig.suptitle("The same noise inputs through training (rows = iteration)", x=0.13, ha="left", fontweight="semibold", fontsize=11, y=0.98)
    save(fig, "17_training_filmstrip.png")


def fig_circuit_summary():
    records = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(DATA, "circuits", "c??.json")))]
    maps = {r["id"]: np.load(os.path.join(DATA, "circuits", f"{r['id']}_map.npy")) for r in records}
    out = os.path.join(OUT, "circuits")
    table = pd.DataFrame([dict(id=r["id"], configuration=r["title"], used_in=r["used"], qubits=r["n_qubits"], trainable_parameters=r["n_params"],
                               gates=r["n_gates"], two_qubit_gates=r["n_two_qubit_gates"], depth=r["depth"], two_qubit_depth=r["two_qubit_depth"],
                               fake_torino_cz=r.get("torino", {}).get("n_two_qubit_gates"), fake_torino_depth=r.get("torino", {}).get("depth"),
                               fake_torino_two_qubit_depth=r.get("torino", {}).get("two_qubit_depth")) for r in records])
    table.to_csv(os.path.join(out, "circuit_resources.csv"), index=False)

    # table
    short = [(r["id"], r["title"].split(", trained")[0]) for r in records]
    fmt = lambda v: "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{int(v):,}"  # noqa: E731
    cells = [[i, t, r["n_qubits"], fmt(r["n_params"]), fmt(r["n_gates"]), fmt(r["n_two_qubit_gates"]), fmt(r["depth"]), fmt(r["two_qubit_depth"]),
              fmt(r.get("torino", {}).get("n_two_qubit_gates")), fmt(r.get("torino", {}).get("depth"))] for (i, t), r in zip(short, records)]
    columns = ["ID", "Configuration", "Qubits", "Params", "Gates", "2-qubit\ngates", "Depth", "2-qubit\ndepth", "CZ on\nfake_torino", "Depth on\nfake_torino"]
    fig, ax = plt.subplots(figsize=(13.5, 0.3 * len(cells) + 0.6))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02)
    ax.axis("off")
    tab = ax.table(cellText=cells, colLabels=columns, bbox=[0, 0, 1, 1], cellLoc="right",
                   colWidths=[0.04, 0.46, 0.05, 0.06, 0.055, 0.065, 0.055, 0.065, 0.075, 0.075])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    for (row, col), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.5)
        cell.set_facecolor(SURFACE)
        cell.PAD = 0.03
        if col in (0, 1):
            cell.set_text_props(ha="left")
        if row == 0:
            cell.set_text_props(fontweight="semibold", color=INK_2)
            cell.set_height(cell.get_height() * 1.5)
    fig.suptitle("Circuit size of every tested configuration", x=0.01, ha="left", fontweight="semibold", fontsize=11, y=0.995)
    save(fig, os.path.join("circuits", "c00_circuit_resources_table.png"),
         "Depth = longest gate path in the exported circuit, without measurements. A CRY counts as one two-qubit gate here (two CNOTs on "
         "hardware). c08-c11 are compiled forms of c01, c06 and c07. K changes the classical noise parameters, not the gates, so c06 and c07 "
         "have the same logical circuit structure.")

    # depth map, every moment of every circuit on one x-scale
    cmap = ListedColormap([SURFACE, "#9ec5f4", ORANGE])
    heights = [m.shape[0] + 1.6 for m in maps.values()]
    fig, axs = plt.subplots(len(records), 1, figsize=(13, 0.2 * sum(heights) + 0.8), sharex=True, gridspec_kw=dict(height_ratios=heights, hspace=0.45))
    fig.subplots_adjust(top=0.94, bottom=0.05)
    x_max = max(m.shape[1] for m in maps.values())
    for ax, r in zip(axs, records):
        m = maps[r["id"]]
        ax.imshow(m, cmap=cmap, vmin=0, vmax=2, aspect="auto", interpolation="nearest", extent=(0, m.shape[1], m.shape[0], 0))
        ax.set_xlim(0, x_max)
        ax.set_yticks([]), ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.text(0, 1.04, f"{r['id']}  {r['title'].split(', trained')[0]}  —  depth {r['depth']}", transform=ax.transAxes, fontsize=8, va="bottom", color=INK)
    axs[-1].set_xlabel("Circuit moment (gate layer)")
    axs[-1].spines["bottom"].set_visible(True)
    axs[-1].tick_params(axis="x", length=3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ["#9ec5f4", ORANGE]]
    fig.legend(handles, ["Single-qubit gate", "Two-qubit gate"], loc="upper right", ncols=2, fontsize=8.5, bbox_to_anchor=(0.91, 0.99))
    fig.suptitle("Full depth of every circuit, one row per qubit, same horizontal scale", x=0.125, ha="left", fontweight="semibold", fontsize=11, y=0.985)
    save(fig, os.path.join("circuits", "c00_circuit_depth_map.png"))


# ------------------------------------------------------------------ decoder comparison, depth (2,1)
DECODERS = {"FRQI_msb": ("FRQI_msb (current)", BLUE), "FRQI": ("FRQI (color qubit)", ORANGE)}


def comparison():
    runs = []
    for path in sorted(glob.glob(os.path.join(DATA, "cmp_*.json"))):
        with open(path) as f:
            run = json.load(f)
        run["df"] = pd.DataFrame(run["rows"])
        run["arrays"] = np.load(path.replace(".json", ".npz"))
        runs.append(run)
    return runs


def fig_decoder_curves(data):
    runs = comparison()
    baseline = data["K = 10"]["df"].iloc[-1]
    refs = data["K = 10"]["refs"]
    panels = [("n_classes_covered", "Digits generated (of 10)", None, None), ("class_tv", "Class distribution distance", None, None),
              ("c2st_blur_auc", "Real vs generated AUC (0.5 is ideal)", refs["c2st_blur_auc"], "real vs real"),
              ("tstr_acc", "Digit accuracy on real images", refs["digit_accuracy_10k"], "trained on real"),
              ("tstr_auc", "Digit macro AUC on real images", refs["digit_auc_10k"], "trained on real"),
              ("mean_image_mae", "Mean image error (MAE, log scale)", refs["real_mean_image_mae"], "real vs real")]
    fig, axs = plt.subplots(2, 3, figsize=(13, 6.2))
    axs[1, 2].set_yscale("log")
    for ax, (column, title, ref, ref_label) in zip(axs.flat, panels):
        for decoding, (label, color) in DECODERS.items():
            group = [r["df"] for r in runs if r["decoding"] == decoding]
            for df in group:
                ax.plot(df["epoch"], df[column], color=color, lw=0.7, alpha=0.45)
            mean = pd.concat([df.set_index("epoch")[column] for df in group], axis=1).mean(axis=1)
            ax.plot(mean.index, mean.values, color=color, lw=2.2, label=label)
        ax.axhline(baseline[column], color=INK_2, lw=0.9)
        ax.text(1.0, baseline[column], " (8,2) baseline", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=INK_2)
        if ref is not None:
            ax.axhline(ref, color=AXIS, lw=0.9)
            ax.text(1.0, ref, f" {ref_label}", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=MUTED)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("Training iteration")
    axs[0, 0].set_ylim(0, 10.8)
    axs[0, 1].set_ylim(bottom=0)
    axs[1, 0].set_ylim(bottom=0)
    axs[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Decoder comparison on the small circuit: depth (2,1), 5 qubits, 28 two-qubit gates", x=0.01, ha="left", fontweight="semibold", fontsize=11)
    fig.tight_layout()
    save(fig, "18_decoder_comparison_curves.png",
         "Thin lines: 3 seeds per decoder, paired (same seed = same initial weights and noise). Thick: mean. '(8,2) baseline' is the K = 10 run at "
         "iteration 10,000. Same metrics, samples and references as figures 06, 10, 11 and 12, so values are directly comparable.")


def fig_decoder_samples(data):
    runs = [r for r in comparison() if r["seed"] == 42]
    first = next(iter(data.values()))["arrays"]
    grids = [("Real MNIST 4x4 (area)", first["real_grid"], first["real_grid_labels"])]
    for decoding, (label, _) in DECODERS.items():
        run = next(r for r in runs if r["decoding"] == decoding)
        grids.append((f"{label}, depth (2,1)", run["arrays"]["grid"], run["arrays"]["grid_labels"]))
    fig = plt.figure(figsize=(13, 5.0))
    outer = fig.add_gridspec(1, 3, wspace=0.12)
    for g, (title, imgs, labels) in enumerate(grids):
        inner = outer[g].subgridspec(10, 11, wspace=0.06, hspace=0.06, width_ratios=[0.6] + [1] * 10)
        for digit in range(10):
            ax = fig.add_subplot(inner[digit, 0])
            ax.axis("off")
            ax.text(0.5, 0.5, str(digit), ha="center", va="center", fontsize=8, color=INK_2)
            row = imgs[labels == digit][:10]
            for c in range(10):
                ax = fig.add_subplot(inner[digit, c + 1])
                show_image(ax, row[c]) if c < len(row) else ax.axis("off")
        fig.text(outer[g].get_position(fig).x0, 0.93, title, fontsize=10, fontweight="semibold", ha="left")
    save(fig, "19_decoder_comparison_samples.png",
         "Seed 42, iteration 10,000. Rows = digit: true label for real images, classifier prediction for generated ones. Empty cells: fewer than 10 samples classified as that digit.")


def fig_decoder_hardware(data):
    runs = comparison()
    with open(os.path.join(DATA, "B.json")) as f:
        base = json.load(f)["K10"]
    b = np.load(os.path.join(DATA, "B.npz"))
    base_mae, base_fid = float(np.abs(b["K10_img_fake_torino"] - b["K10_img_ideal"]).mean()), float(b["K10_fid_fake_torino"].mean())
    base_aer_mae = float(np.abs(b["K10_img_aer"] - b["K10_img_ideal"]).mean())
    rows = []
    for decoding, (label, _) in DECODERS.items():
        run = next(r for r in runs if r["decoding"] == decoding and r["seed"] == 42)
        rows += [(run, "ideal", f"{label}\nsimulation"), (run, "fake_torino", f"{label}\nfake_torino, 4096 shots")]
    n = 10
    fig, axs = plt.subplots(len(rows), n + 1, figsize=(1.0 * n + 2.6, 0.92 * len(rows) + 0.3),
                            gridspec_kw=dict(width_ratios=[2.6] + [1] * n, wspace=0.08, hspace=0.12))
    fig.subplots_adjust(top=0.9)
    for r, (run, key, label) in enumerate(rows):
        axs[r, 0].axis("off")
        axs[r, 0].text(1.0, 0.5, label, ha="right", va="center", fontsize=8.5, transform=axs[r, 0].transAxes)
        for c in range(n):
            show_image(axs[r, c + 1], run["arrays"][f"hw_{key}"][c])
    fig.suptitle("The small circuit through simulated device noise (seed 42, iteration 10,000)", x=0.01, ha="left", fontweight="semibold", fontsize=11, y=1.0)
    save(fig, "20_decoder_comparison_device_images.png", "Same 10 noise inputs per column within each decoder pair.")

    metrics = [("fake_torino", "fidelity", "Fidelity to simulation", base_fid), ("aer", "pixel_mae", "Pixel error, Aer (shot noise only)", base_aer_mae),
               ("fake_torino", "pixel_mae", "Pixel error, fake_torino", base_mae), ("fake_torino", "confidence", "Digit classifier confidence", None),
               ("fake_torino", "n_classes_covered", "Digits generated (of 10)", None)]
    fig, axs = plt.subplots(1, 5, figsize=(14.5, 3.0))
    for ax, (backend, key, title, reference) in zip(axs, metrics):
        for x, (decoding, (label, color)) in enumerate(DECODERS.items()):
            group = [r["hardware"][backend][key] for r in runs if r["decoding"] == decoding]
            ax.bar(x, np.mean(group), 0.55, color=color)
            ax.scatter([x] * len(group), group, s=18, color=INK, zorder=3)
            ax.annotate(f"{np.mean(group):.3f}" if key != "n_classes_covered" else f"{np.mean(group):.1f}", (x, np.max(group)), xytext=(0, 6),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.08)
        if reference is not None:
            ax.axhline(reference, color=INK_2, lw=0.9)
            ax.text(1.0, reference, " (8,2)\n baseline", transform=ax.get_yaxis_transform(), va="center", fontsize=7.5, color=INK_2)
        ax.set_xticks([0, 1], [v[0].split(" ")[0] for v in DECODERS.values()])
        ax.grid(axis="x", visible=False)
        ax.set_title(title, fontsize=9.5)
    axs[4].set_ylim(0, 10.8)
    fig.suptitle("Small circuit on simulated devices, 4096 shots", x=0.01, ha="left", fontweight="semibold", fontsize=11)
    fig.tight_layout()
    cz = np.mean([r["hardware"]["fake_torino"]["n_two_qubit_gates_mean"] for r in runs])
    save(fig, "21_decoder_comparison_device_metrics.png",
         f"100 noise inputs per run. Bars: mean of 3 seeds, dots: each seed. About {cz:.0f} CZ gates after routing, against {base['fake_torino']['n_two_qubit_gates']} for the (8,2) baseline "
         "(baseline lines: K = 10 run, 20 noise inputs). Confidence and digit counts here come from only 100 images each.")

    # numbers table
    table = []
    for run in runs:
        last = run["df"].iloc[-1]
        hw = run["hardware"]["fake_torino"]
        table.append(dict(decoding=run["decoding"], seed=run["seed"], model=run["model"], digits=last["n_classes_covered"], class_tv=last["class_tv"],
                          c2st_auc=last["c2st_blur_auc"], c2st_acc=last["c2st_blur_acc"], critic_auc=last["critic_auc"], digit_acc=last["tstr_acc"], digit_auc=last["tstr_auc"],
                          confidence=last["confidence"], mean_image_mae=last["mean_image_mae"], std_image_mae=last["std_image_mae"], marginal_w1=last["marginal_w1"],
                          cov_rel_error=last["cov_rel_error"], fake_to_real_rmse=last["fake_to_real_rmse"], real_to_fake_rmse=last["real_to_fake_rmse"],
                          torino_fidelity=hw["fidelity"], torino_pixel_mae=hw["pixel_mae"], torino_cz=hw["n_two_qubit_gates_mean"], aer_pixel_mae=run["hardware"]["aer"]["pixel_mae"],
                          aer_fidelity=run["hardware"]["aer"]["fidelity"], torino_confidence=hw["confidence"], torino_digits=hw["n_classes_covered"], ideal_confidence_100=run["hardware"]["ideal"]["confidence"],
                          **{f"late_{column}": late_mean(run, column) for column, _, _ in PAIRED}))
    pd.DataFrame(table).sort_values(["seed", "decoding"]).to_csv(os.path.join(OUT, "decoder_comparison.csv"), index=False)
    print("wrote decoder_comparison.csv", flush=True)


LATE = list(range(6000, 10001, 1000))
PAIRED = [("n_classes_covered", "Digits generated (of 10)", 1), ("class_tv", "Class distribution distance", -1),
          ("c2st_blur_auc", "Real vs generated AUC", -1), ("tstr_acc", "Digit accuracy on real images", 1),
          ("tstr_auc", "Digit macro AUC on real images", 1), ("mean_image_mae", "Mean image error", -1)]
HW_PAIRED = [("fidelity", "fake_torino fidelity", 1), ("pixel_mae", "fake_torino pixel error", -1)]


def late_mean(run, column):
    df = run["df"]
    return float(df[df["epoch"].isin(LATE)][column].mean())


def fig_decoder_paired(data):
    runs = comparison()
    by = {(r["decoding"], r["seed"]): r for r in runs}
    seeds = sorted({r["seed"] for r in runs})
    panels = [(t, s, lambda r, c=c: late_mean(r, c), c in ("n_classes_covered",)) for c, t, s in PAIRED]
    panels += [(t, s, lambda r, c=c: r["hardware"]["fake_torino"][c], False) for c, t, s in HW_PAIRED]
    fig, axs = plt.subplots(len(panels), 1, figsize=(9.5, 0.62 * len(panels) + 1.0))
    for ax, (title, sign, value, is_count) in zip(axs, panels):
        frqi, msb = np.array([value(by[("FRQI", s)]) for s in seeds]), np.array([value(by[("FRQI_msb", s)]) for s in seeds])
        advantage = sign * (frqi - msb)
        limit = 1.3 * max(np.abs(advantage).max(), 1e-9)
        ax.axvspan(0, limit, color=ORANGE, alpha=0.07, lw=0)
        ax.axvspan(-limit, 0, color=BLUE, alpha=0.07, lw=0)
        ax.axvline(0, color=INK_2, lw=0.9)
        ax.scatter(advantage, np.zeros(len(seeds)), s=30, color=INK, zorder=3)
        previous = None
        for a, s in sorted(zip(advantage, seeds)):  # neighbours closer than 6% of the axis get their label below
            below = previous is not None and a - previous < 0.12 * limit
            ax.annotate(str(s), (a, 0), xytext=(0, -13 if below else 6), textcoords="offset points", ha="center", fontsize=7, color=INK_2)
            previous = None if below else a
        ax.plot([advantage.mean()] * 2, [-0.45, 0.45], color=INK, lw=2.4)
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-1, 1)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="x", labelsize=7.5)
        fmt = "{:.1f}" if is_count else "{:.3f}"
        ax.text(-0.01, 0.5, title, transform=ax.transAxes, ha="right", va="center", fontsize=8.5)
        ax.text(1.01, 0.5, f"FRQI {fmt.format(frqi.mean())}  vs  FRQI_msb {fmt.format(msb.mean())}", transform=ax.transAxes,
                ha="left", va="center", fontsize=8, color=INK_2)
        print(f"{title:32s} FRQI {frqi.round(4)} msb {msb.round(4)} advantage {advantage.round(4)}", flush=True)
    axs[0].text(0.25, 1.25, "← FRQI_msb better", transform=axs[0].transAxes, ha="center", fontsize=8.5, color=BLUE)
    axs[0].text(0.75, 1.25, "FRQI better →", transform=axs[0].transAxes, ha="center", fontsize=8.5, color=ORANGE)
    fig.subplots_adjust(hspace=0.9, top=0.88)
    fig.suptitle("Paired difference per seed, FRQI against FRQI_msb (depth (2,1))", x=0.01, ha="left", fontweight="semibold", fontsize=11, y=0.97)
    save(fig, "22_decoder_comparison_paired.png",
         "Dots: one seed each (numbered), same initial weights for both decoders. Bar: mean of the 3 seeds. Each row has its own scale. "
         "Top six rows: average of checkpoints 6,000 to 10,000 (5 checkpoints, 10,000 samples each), to reduce checkpoint-to-checkpoint swings. "
         "Bottom two: final checkpoint, 100 noise inputs, 4096 shots. Right: means over seeds. With 3 seeds, none of these differences is statistically established.")


if __name__ == "__main__":
    needs_diag = dict(progress=fig_progress, samples=fig_samples, loss=fig_loss, separability=fig_separability, decoder_curves=fig_decoder_curves,
                      decoder_samples=fig_decoder_samples, decoder_hardware=fig_decoder_hardware, decoder_paired=fig_decoder_paired,
                      digits=fig_digit_task, pixels=fig_pixel_curves, maps=fig_pixel_maps, perdigit=fig_per_digit,
                      film=fig_filmstrip)
    plain = dict(trained=fig_trained_export, shots=fig_shot_noise, training=fig_training_checks, compiled=fig_compiled_step,
                 mmd=fig_mmd, hardware=fig_hardware_error, circuits=fig_circuit_summary)
    names = sys.argv[1:] or [*plain, *needs_diag]
    data = diag() if any(n in needs_diag for n in names) else None
    for name in names:
        if name == "loss":
            fig_loss(data)
        elif name in needs_diag:
            needs_diag[name](data)
        else:
            plain[name]()
