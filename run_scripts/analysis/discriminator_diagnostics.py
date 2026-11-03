"""Plot discriminator training diagnostics for every experiment run.

Reads ``evaluation_summary.csv`` from each model folder and produces:

* one loss/gradient figure per run group,
* an overview figure covering every run,
* ``discriminator_run_summary.csv`` with start/end values per run.

Two caveats apply to the logged ``loss_discriminator`` column and are repeated
in the figure captions:

1. it includes the WGAN-GP gradient-penalty term, so it is not the pure
   Wasserstein estimate, and
2. only the last of the ``discriminator_training_steps`` inner updates is
   recorded each iteration, so it is a snapshot rather than an average.

Usage::

    python run_scripts/analysis/discriminator_diagnostics.py \
        --experiments-dir experiments --out-dir plots_discriminator
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

CAPTION = (
    "Logged L_D includes the gradient-penalty term and records only the last "
    "inner discriminator step of each iteration."
)


def load_runs(experiments_dir: Path):
    """Return a list of (name, dataframe, metadata) for every run with metrics."""
    runs = []
    for folder in sorted(p for p in experiments_dir.iterdir() if p.is_dir()):
        summary = folder / "evaluation_summary.csv"
        if not summary.exists():
            continue
        meta_file = folder / "meta.json"
        meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
        runs.append((folder.name, pd.read_csv(summary), meta))
    return runs


def smooth(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window, center=True, min_periods=1).mean()


def dataset_of(name: str) -> str:
    for key in ("fashion", "svhn", "mnist"):
        if key in name:
            return key
    return "other"


def plot_run(name, df, meta, out_dir: Path, window: int) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    for column, color, label in (
        ("loss_generator", "tab:blue", "Generator $L_G$"),
        ("loss_discriminator", "tab:orange", "Discriminator $L_D$"),
    ):
        if column in df:
            axes[0].plot(df.iteration, df[column], lw=0.3, alpha=0.3, color=color)
            axes[0].plot(df.iteration, smooth(df[column], window), color=color, label=label)
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"{name}\ndepth {meta.get('circuit_depth')}, modes {meta.get('noise_tuning')}", fontsize=9)

    for column, color, label in (
        ("grad_mag_generator", "tab:blue", "Generator"),
        ("grad_mag_discriminator", "tab:orange", "Discriminator"),
    ):
        if column in df:
            axes[1].plot(df.iteration, smooth(df[column], window), color=color, label=label)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Gradient magnitude")
    axes[1].legend(fontsize=8)

    fig.text(0.01, 0.005, CAPTION, fontsize=6.5, color="0.35")
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(out_dir / f"run_{name[:60]}.png", dpi=150)
    plt.close(fig)


def plot_overview(runs, out_dir: Path, window: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, df, _meta in runs:
        if "loss_discriminator" in df:
            axes[0].plot(df.iteration, smooth(df.loss_discriminator, window), lw=1)
        if "grad_mag_discriminator" in df:
            axes[1].plot(df.iteration, smooth(df.grad_mag_discriminator, window), lw=1)
    axes[0].set_title(f"Discriminator loss, all {len(runs)} runs")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("$L_D$ (includes gradient penalty)")
    axes[1].set_title("Discriminator gradient magnitude, all runs")
    axes[1].set_xlabel("Iteration")
    axes[1].set_yscale("log")
    fig.text(0.01, 0.005, CAPTION, fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_dir / "overview_all_runs.png", dpi=150)
    plt.close(fig)


def summarize(runs, tail: int) -> pd.DataFrame:
    rows = []
    for name, df, meta in runs:
        last = df.tail(tail)
        first = df.head(min(200, len(df)))
        row = {
            "run": name,
            "dataset": dataset_of(name),
            "depth": str(meta.get("circuit_depth")),
            "modes": meta.get("noise_tuning"),
            "iterations": int(df.iteration.max()) if "iteration" in df else None,
            "LD_start": round(first.loss_discriminator.mean(), 3) if "loss_discriminator" in df else None,
            "LD_end": round(last.loss_discriminator.mean(), 3) if "loss_discriminator" in df else None,
            "D_grad_end": round(last.grad_mag_discriminator.mean(), 3) if "grad_mag_discriminator" in df else None,
        }
        for column in df.columns:
            if column.startswith("grad_norm_interpolates"):
                values = df[column].dropna()
                row["interp_grad_norm_end"] = round(values.tail(20).mean(), 3) if len(values) else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["dataset", "depth"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", default="experiments", type=Path)
    parser.add_argument("--out-dir", default="plots_discriminator", type=Path)
    parser.add_argument("--window", default=500, type=int, help="smoothing window in iterations")
    parser.add_argument("--tail", default=2000, type=int, help="iterations averaged for the end-of-run value")
    parser.add_argument("--per-run-plots", action="store_true", help="also write one figure per run")
    args = parser.parse_args()

    runs = load_runs(args.experiments_dir)
    if not runs:
        raise SystemExit(f"no evaluation_summary.csv found under {args.experiments_dir}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_overview(runs, args.out_dir, args.window)
    if args.per_run_plots:
        for name, df, meta in runs:
            plot_run(name, df, meta, args.out_dir, args.window)

    table = summarize(runs, args.tail)
    table.to_csv(args.out_dir / "discriminator_run_summary.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nwrote figures and discriminator_run_summary.csv to {args.out_dir}")


if __name__ == "__main__":
    main()
