"""Figures of the 0/1 depth sweep. Renders from plotdata/ (written by collect_run.py) and each run's evaluation_summary.csv.

Usage: python make_plots.py [runs csv] [experiments root] [output folder]
Defaults: experiments_configs/runs_01_depth_sweep.csv, apps/logistics/experiments, plots_01_depth_sweep/.
"""
import ast
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "experiments_configs", "runs_01_depth_sweep.csv")
EXP_ROOT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, "apps", "logistics", "experiments")
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(REPO, "plots_01_depth_sweep")
DATA = os.environ.get("PLOTDATA_DIR", os.path.join(HERE, "plotdata"))
os.makedirs(OUT, exist_ok=True)

# Two-qubit gates after compiling for fake_torino (CZ count), FRQI color rotation next-neighbor generator, 5 qubits.
# Computed for this sweep with qiskit_export + ibm.transpile, same procedure as plot_generation/draw_circuits.py.
TORINO_CZ = {(2, 1): 57, (2, 2): 90, (3, 1): 97, (4, 1): 130, (3, 2): 140, (4, 2): 186, (6, 2): 283, (8, 2): 378}
# Categorical slots 1-4 (validated default palette), assigned in depth order, never cycled
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#1f1f1e", "#5f5e59", "#e6e5e0"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "lines.linewidth": 2, "legend.frameon": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

# ---- load runs
cfg = pd.read_csv(CSV, dtype=str, keep_default_na=False)
runs = []
for _, r in cfg.iterrows():
    name = os.path.basename(r["result_path"].strip())
    path = os.path.join(DATA, f"{name}.json")
    if not name.startswith("continuous_") or not os.path.exists(path):
        print("skipping (not finished or not collected):", r["result_path"][:80], r["circuit_depth"], r["random_seed"])
        continue
    with open(path) as f:
        j = json.load(f)
    runs.append(dict(name=name, depth=tuple(ast.literal_eval(r["circuit_depth"])), seed=int(r["random_seed"]),
                     rows=pd.DataFrame(j["rows"]), ref=j["real_ref"], arrays=dict(np.load(os.path.join(DATA, f"{name}.npz"))),
                     losses=pd.read_csv(os.path.join(EXP_ROOT, name, "evaluation_summary.csv"))))
assert runs, "no collected runs"
depths = sorted({r["depth"] for r in runs}, key=lambda d: (TORINO_CZ.get(d, 1e9), d))
color = {d: COLORS[i] for i, d in enumerate(depths)}
seeds = sorted({r["seed"] for r in runs})
ref = runs[0]["ref"]


def label(d):
    return f"depth {d}  ({TORINO_CZ.get(d, '?')} CZ)"


def by_depth(d):
    return [r for r in runs if r["depth"] == d]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


def show(ax, img, vmax=1.0):
    ax.imshow(np.asarray(img).reshape(4, 4), cmap="gray", vmin=0, vmax=vmax)
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---- 01: quality over training, one panel per metric (mean line over seeds, band = range of seeds)
metrics = [("c2st_auc", "Real-vs-generated AUC", "AUC (lower is better)", ref["c2st_auc"], "real vs real"),
           ("fake_to_real_rmse", "Realism: distance to nearest real image", "RMS pixel error (lower is better)",
            None, None),
           ("real_to_fake_rmse", "Coverage: real image to nearest generated", "RMS pixel error (lower is better)",
            ref["nn_rmse"], "real vs real"),
           ("share_1", "Share of generated images that look like a 1", "fraction", ref["share_1"], "real data")]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, (key, title, ylabel, refval, reflabel) in zip(axes.ravel(), metrics):
    for d in depths:
        tables = [r["rows"].set_index("epoch")[key] for r in by_depth(d)]
        t = pd.concat(tables, axis=1)
        ax.plot(t.index, t.mean(axis=1), color=color[d], label=label(d))
        if t.shape[1] > 1:
            ax.fill_between(t.index, t.min(axis=1), t.max(axis=1), color=color[d], alpha=0.15, linewidth=0)
    if refval is not None:
        ax.axhline(refval, color=INK2, linestyle="--", linewidth=1.2)
        below = key == "real_to_fake_rmse"  # generated curves sit just above this line; label it from underneath
        if below:
            ax.set_ylim(bottom=0)
        ax.annotate(reflabel, (ax.get_xlim()[1], refval), xytext=(-4, -11 if below else 4), textcoords="offset points",
                    ha="right", color=INK2, fontsize=8)
    ax.set_title(title, loc="left", color=INK, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel("training iteration")
axes[0, 0].legend(loc="upper right", fontsize=8)
fig.suptitle(f"0/1 MNIST 4x4: generation quality by circuit depth (line = mean of seeds {seeds}, band = range)",
             x=0.01, ha="left", color=INK)
fig.text(0.01, -0.01, "AUC: how well a classifier tells real from generated images; 0.5 means it can't. "
         "Realism: are generated images close to some real image? Coverage: is every kind of real image generated?",
         fontsize=8, color=INK2)
fig.tight_layout()
save(fig, "01_quality_curves.png")

# ---- 02: final quality vs hardware cost
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, key, title in [(axes[0], "c2st_auc", "Real-vs-generated AUC (lower is better)"),
                       (axes[1], "real_to_fake_rmse", "Coverage error (lower is better)")]:
    for d in depths:
        vals = [r["rows"].iloc[-1][key] for r in by_depth(d)]
        x = TORINO_CZ.get(d, np.nan)
        ax.scatter([x] * len(vals), vals, s=36, color=color[d], edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(str(d), (x, np.mean(vals)), xytext=(7, 0), textcoords="offset points", va="center", fontsize=9,
                    color=INK)
    m = [np.mean([r["rows"].iloc[-1][key] for r in by_depth(d)]) for d in depths]
    ax.plot([TORINO_CZ.get(d, np.nan) for d in depths], m, color=INK2, linewidth=1, zorder=2)
    ax.set_xlabel("two-qubit gates on fake_torino (CZ) - hardware noise grows with this")
    ax.set_title(title, loc="left", color=INK, fontweight="bold")
fig.suptitle("Quality at the last checkpoint vs. hardware cost (one dot per seed)", x=0.01, ha="left", color=INK)
fig.tight_layout()
save(fig, "02_quality_vs_hardware_cost.png")

# ---- 03: final samples per depth (first seed), real row on top
seed0 = seeds[0]
rows_ = [("real", runs[0]["arrays"]["real_grid"])] + \
        [(label(d), next(r for r in by_depth(d) if r["seed"] == seed0)["arrays"]["grid"]) for d in depths
         if any(r["seed"] == seed0 for r in by_depth(d))]
fig, axes = plt.subplots(len(rows_), 16, figsize=(12, 0.95 * len(rows_) + 0.4))
axes = np.atleast_2d(axes)
for i, (name, imgs) in enumerate(rows_):
    for j in range(16):
        ax = axes[i, j]
        if j < len(imgs):
            show(ax, imgs[j])
        else:
            ax.axis("off")
    axes[i, 0].set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8, color=INK)
fig.suptitle(f"Final samples (seed {seed0}): left 8 look like 0, right 8 look like 1 (sorted by a real-data classifier)",
             x=0.01, ha="left", color=INK, fontsize=10)
save(fig, "03_samples_by_depth.png")

# ---- 04: mean image per digit, real vs each depth (first seed)
fig, axes = plt.subplots(2, len(depths) + 1, figsize=(2 * (len(depths) + 1), 5.0))
axes = np.atleast_2d(axes)
for k in (0, 1):
    show(axes[k, 0], runs[0]["arrays"][f"real_mean_{k}"])
    axes[k, 0].set_ylabel(f"digit {k}", color=INK)
    axes[0, 0].set_title("real", color=INK, fontsize=9)
    for i, d in enumerate(depths):
        r = next((r for r in by_depth(d) if r["seed"] == seed0), by_depth(d)[0])
        show(axes[k, i + 1], r["arrays"][f"fake_mean_{k}"])
        mae = r["rows"].iloc[-1].get(f"digit_mae_{k}", np.nan)
        axes[k, i + 1].set_xlabel("none generated" if pd.isna(mae) else f"error {mae:.3f}", fontsize=8, color=INK2)
        if k == 0:
            axes[0, i + 1].set_title(f"{d}", color=INK, fontsize=9)
fig.suptitle("Average generated 0 and 1 vs. the real average (error = mean absolute pixel difference)",
             x=0.01, ha="left", color=INK, fontsize=10)
fig.tight_layout(h_pad=2.0)
save(fig, "04_mean_digits.png")

# ---- 05: training filmstrip, first sample of each checkpoint row per depth (first seed)
film_epochs = sorted({int(k.split("_")[1]) for k in runs[0]["arrays"] if k.startswith("film_")})
show_epochs = [e for e in [0, 100, 500, 1000, 2000, 5000, 10000] if e in film_epochs] or film_epochs
fig, axes = plt.subplots(len(depths), len(show_epochs) * 4, figsize=(1.0 * len(show_epochs) * 4 * 0.42, 0.55 * len(depths) + 0.8))
axes = np.atleast_2d(axes)
for i, d in enumerate(depths):
    r = next((r for r in by_depth(d) if r["seed"] == seed0), by_depth(d)[0])
    for e_i, e in enumerate(show_epochs):
        for s in range(4):
            show(axes[i, e_i * 4 + s], r["arrays"].get(f"film_{e}", np.full(16, np.nan))[s])
            if i == 0 and s == 0:
                axes[0, e_i * 4].set_title(f"it {e}", loc="left", fontsize=8, color=INK)
    axes[i, 0].set_ylabel(f"{d}", rotation=0, ha="right", va="center", fontsize=8, color=INK)
fig.suptitle(f"Training progress (seed {seed0}, same noise inputs at every checkpoint)", x=0.01, ha="left", color=INK,
             fontsize=10)
save(fig, "05_training_filmstrip.png")

# ---- 06: losses (two panels, one scale each), smoothed over 200 iterations
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, key, title in [(axes[0], "loss_generator", "Generator loss"), (axes[1], "loss_discriminator", "Critic loss")]:
    for d in depths:
        t = pd.concat([r["losses"].set_index("iteration")[key] for r in by_depth(d)], axis=1).mean(axis=1)
        ax.plot(t.index, t.rolling(200, min_periods=1).mean(), color=color[d], label=label(d))
    ax.set_title(title + " (mean of seeds, 200-iteration rolling mean)", loc="left", color=INK, fontweight="bold")
    ax.set_xlabel("training iteration")
axes[0].legend(fontsize=8)
fig.tight_layout()
save(fig, "06_loss_curves.png")

# ---- summary table
summary = pd.DataFrame([dict(depth=str(r["depth"]), seed=r["seed"], torino_cz=TORINO_CZ.get(r["depth"]),
                             last_iteration=int(r["rows"].iloc[-1]["epoch"]),
                             **{k: round(float(r["rows"].iloc[-1][k]), 4) for k in
                                ["c2st_auc", "fake_to_real_rmse", "real_to_fake_rmse", "share_1", "mean_image_mae"]},
                             run=r["name"]) for r in runs]).sort_values(["torino_cz", "seed"])
summary.to_csv(os.path.join(OUT, "summary.csv"), index=False)
print(summary.drop(columns="run").to_string(index=False))
