"""Score real images against generated ones with a trained critic.

A GAN has no class labels, so a confusion matrix does not apply directly. The
closest equivalent is to score a batch of real images and a batch of generated
images with the trained discriminator and compare the two distributions.

Under the WGAN formulation the critic is *meant* to separate the two: the gap
between the mean scores is its estimate of the Wasserstein distance. The useful
signal is therefore the size of that gap and how it shrinks across checkpoints,
not whether the two piles overlap at any single checkpoint.

The script also recomputes ``||grad_x D||`` on interpolated real/fake images.
That is the quantity the WGAN-GP penalty constrains towards 1, and it is
discarded during training unless the handler patch that logs it is applied.

Fake images come either from a ``samples_iteration=*.npy`` file saved during
training (default, no quantum simulation needed) or from re-running the
generator with ``--generate`` (slower, requires the full training dependencies).

Usage::

    python run_scripts/analysis/discriminator_real_vs_fake.py \
        --run experiments/<model folder> \
        --iteration 20000 \
        --real-data apps/logistics/training_data/<dataset>.npy \
        --out-dir plots_discriminator
"""

import argparse
import pickle
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from qugen.main.discriminator.cnn_discriminator import ConvDiscriminator  # noqa: E402


def latest_iteration(run: Path) -> int:
    iterations = [
        int(match.group(1))
        for path in run.glob("parameters_training_iteration=*.pickle")
        if (match := re.search(r"iteration=(\d+)", path.name))
    ]
    if not iterations:
        raise SystemExit(f"no parameter checkpoints in {run}")
    return max(iterations)


def load_weights(run: Path, iteration: int):
    """Checkpoints store a (generator_weights, discriminator_weights) tuple."""
    path = run / f"parameters_training_iteration={iteration}.pickle"
    with path.open("rb") as handle:
        generator_weights, discriminator_weights = pickle.load(handle)
    return generator_weights, discriminator_weights


def load_real(path: Path, n_samples: int, seed: int) -> np.ndarray:
    data = np.load(path)
    data = data.reshape(len(data), -1).astype(np.float32)
    if data.max() > 1.0:  # training divides raw pixel values by 255
        data = data / 255.0
    rng = np.random.default_rng(seed)
    index = rng.choice(len(data), min(n_samples, len(data)), replace=False)
    return data[index]


def load_fake_from_samples(run: Path, iteration: int) -> np.ndarray:
    path = run / f"samples_iteration={iteration}.npy"
    if not path.exists():
        candidates = sorted(run.glob("samples_iteration=*.npy"))
        if not candidates:
            raise SystemExit(
                f"no samples_iteration=*.npy in {run}; re-run with --generate to sample from the model"
            )
        path = candidates[-1]
    return np.load(path).reshape(-1, np.load(path).shape[-1]).astype(np.float32)


def generate_fake(run: Path, iteration: int, n_samples: int) -> np.ndarray:
    """Re-run the quantum generator. Needs the full training dependencies."""
    from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler

    handler = ContinuousQGANModelHandler()
    handler.reload(run.name, iteration)
    return np.asarray(handler.sample(n_samples=n_samples)).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="experiment folder of a trained model")
    parser.add_argument("--real-data", required=True, type=Path, help="preprocessed .npy training data")
    parser.add_argument("--iteration", type=int, default=None, help="checkpoint iteration (default: latest)")
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--n-channels", type=int, default=1, help="3 for colour datasets such as SVHN")
    parser.add_argument("--generate", action="store_true", help="sample from the generator instead of saved samples")
    parser.add_argument("--out-dir", default=Path("plots_discriminator"), type=Path)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    iteration = args.iteration if args.iteration is not None else latest_iteration(args.run)
    _generator_weights, discriminator_weights = load_weights(args.run, iteration)
    critic = ConvDiscriminator(is_critic=True, n_channels=args.n_channels)

    real = load_real(args.real_data, args.n_samples, args.seed)
    fake = (
        generate_fake(args.run, iteration, args.n_samples)
        if args.generate
        else load_fake_from_samples(args.run, iteration)
    )
    n_pairs = min(len(real), len(fake))
    real, fake = real[:n_pairs], fake[:n_pairs]

    real_scores = np.asarray(critic.apply(discriminator_weights, jnp.asarray(real))).ravel()
    fake_scores = np.asarray(critic.apply(discriminator_weights, jnp.asarray(fake))).ravel()
    gap = float(real_scores.mean() - fake_scores.mean())

    # ||grad_x D|| on points between real and fake, the quantity the penalty constrains.
    grad_fn = jax.vmap(
        jax.grad(lambda weights, image: critic.apply(weights, image[None]).squeeze(), argnums=1),
        in_axes=(None, 0),
    )
    rng = np.random.default_rng(args.seed)
    alpha = rng.uniform(size=(n_pairs, 1)).astype(np.float32)
    interpolates = jnp.asarray(real + alpha * (fake - real))
    slopes = np.asarray(jnp.linalg.norm(grad_fn(discriminator_weights, interpolates), axis=1))

    threshold = (real_scores.mean() + fake_scores.mean()) / 2
    correct = int((real_scores > threshold).sum() + (fake_scores <= threshold).sum())

    print(f"run           : {args.run.name}")
    print(f"iteration     : {iteration}  ({'generated' if args.generate else 'saved'} samples, n={n_pairs})")
    print(f"real score    : mean {real_scores.mean():.3f}  sd {real_scores.std():.3f}")
    print(f"fake score    : mean {fake_scores.mean():.3f}  sd {fake_scores.std():.3f}")
    print(f"gap           : {gap:.3f}   (the critic's Wasserstein estimate)")
    print(f"split at {threshold:.3f}: {correct}/{2 * n_pairs} on the correct side "
          f"({correct / (2 * n_pairs):.1%}; threshold taken from these same scores, so descriptive only)")
    print(f"||grad_x D||  : mean {slopes.mean():.3f}  median {np.median(slopes):.3f}  sd {slopes.std():.3f}  (target 1.0)")
    print(f"penalty term  : 10 * mean((||g|| - 1)^2) = {10 * np.mean((slopes - 1) ** 2):.3f}")
    print(f"=> predicted logged L_D = -gap + penalty = {-gap + 10 * np.mean((slopes - 1) ** 2):.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(real_scores, bins=40, alpha=0.6, label=f"real (mean {real_scores.mean():.2f})")
    axes[0].hist(fake_scores, bins=40, alpha=0.6, label=f"generated (mean {fake_scores.mean():.2f})")
    axes[0].axvline(threshold, color="k", ls="--", lw=1)
    axes[0].set_xlabel("critic score")
    axes[0].set_title(f"Real vs generated critic scores\n{args.run.name[:42]}, iter {iteration}", fontsize=9)
    axes[0].legend(fontsize=8)
    axes[1].hist(slopes, bins=40, color="tab:green")
    axes[1].axvline(1.0, color="k", ls="--")
    axes[1].set_xlabel(r"$\|\nabla_x D\|$ on interpolates")
    axes[1].set_title(f"Gradient-penalty check\nmean {slopes.mean():.2f}, target 1.0", fontsize=9)
    fig.tight_layout()
    out_png = args.out_dir / f"real_vs_fake_{args.run.name[:40]}_iter{iteration}.png"
    fig.savefig(out_png, dpi=150)
    np.savez(
        args.out_dir / f"scores_{args.run.name[:40]}_iter{iteration}.npz",
        real_scores=real_scores,
        fake_scores=fake_scores,
        slopes=slopes,
    )
    print(f"\nwrote {out_png}")


if __name__ == "__main__":
    main()
