"""
Compare a trained QGAN generator under IonQ noise with its exact (noise-free) simulation.

The generator circuits of a trained model are run on the IonQ cloud simulator once per noise profile, e.g., 'ideal' (no
noise, only shot noise) and 'aria-1' (simulated noise of IonQ's Aria device). The estimated output probabilities are
compared with the exact ones of the simulated generator via the classical fidelity and decoded into images, i.e., it
shows how the images of the QGAN would look like if the generator ran on IonQ hardware. The noise only acts when the
images are generated: the models themselves were trained in noise-free simulation.

Needs the optional dependencies (pip install ".[ionq]") and an IonQ API key in the environment variable IONQ_API_KEY
or in a .env file. Run from apps/logistics, e.g.
    python ../../run_scripts/sample_on_ionq.py --model_name=continuous_mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area_minmax_qgan_6538ebe1
    python ../../run_scripts/sample_on_ionq.py --model_name=... --epoch=5000 --n_samples=4 --shots=1000
"""

import os

import matplotlib
import numpy as np
import pandas as pd
from fire import Fire

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler
from qugen.main.hardware import ionq, qiskit_export


def _ionq_runner(shots, seed):
    """Function running circuits on the IonQ simulator with a given noise profile, returning counts and the job ID."""
    def run(circuits, noise_model):
        backend = ionq.get_backend("ionq_simulator", noise_model=noise_model)
        return ionq.run_circuits(circuits, backend, shots=shots, seed=seed, job_tags=["qgan", noise_model])
    return run


def compare(model, run, noise_models=("ideal", "aria-1"), n_samples=4, seed=0, smoothing=0.5):
    """
    Run the generator circuits of a (reloaded) model under each noise model and compare them with the exact output.

    Args:
        model (ContinuousQGANModelHandler): The reloaded model handler.
        run (Callable): Function (circuits, noise_model) -> (counts per circuit, job ID), see _ionq_runner().
        noise_models (tuple[str], optional): Noise profiles to run.
        n_samples (int, optional): Number of noise inputs, i.e., generated images.
        seed (int, optional): Seed for the noise inputs.
        smoothing (float, optional): Pseudo-count per basis state for decoding the images (FRQI cannot decode states
            that were never observed). The fidelities are computed without it.

    Returns:
        tuple[pd.DataFrame, dict[str, np.ndarray]]: Fidelity of each generated image and the images ('exact' and one
        entry per noise model).
    """
    n_qubits = model.n_qubits + model.n_ancilla_qubits
    noise = qiskit_export.sample_noise(model, n_samples, seed=seed)
    circuits = qiskit_export.generator_circuits(model, noise)
    exact_probs = qiskit_export.ideal_probs(model, noise)

    images = {"exact": qiskit_export.decode_probs(model, exact_probs)}
    rows = []
    for noise_model in noise_models:
        counts, job_id = run(circuits, noise_model)
        estimated = np.stack([qiskit_export.counts_to_probs(c, n_qubits) for c in counts])
        smoothed = np.stack([qiskit_export.counts_to_probs(c, n_qubits, smoothing=smoothing) for c in counts])
        images[noise_model] = qiskit_export.decode_probs(model, smoothed)

        for sample, fidelity in enumerate(qiskit_export.classical_fidelity(estimated, exact_probs)):
            rows.append(dict(noise_model=noise_model, sample=sample, fidelity=float(fidelity), job_id=job_id))
    return pd.DataFrame(rows), images


def plot_images(images, n_channels, path):
    """One row of images per entry of images (exact simulation first), one column per generated sample."""
    names = list(images)
    n_samples = len(images[names[0]])
    side = int(round(np.sqrt(images[names[0]].shape[1] / n_channels)))

    fig, axes = plt.subplots(len(names), n_samples, figsize=(1.6 * n_samples + 1.2, 1.7 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        for col in range(n_samples):
            image = np.clip(images[name][col].reshape(side, side, n_channels), 0, 1)
            axes[row][col].imshow(image.squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])
        axes[row][0].set_ylabel("exact (QGAN)" if name == "exact" else f"IonQ {name}", rotation=0, ha="right", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(model_name, epoch=None, n_samples=4, shots=1000, seed=0, noise_models=("ideal", "aria-1"), smoothing=0.5,
         out_dir="ionq_results"):
    if isinstance(noise_models, str):
        noise_models = (noise_models,)

    if epoch is None:
        epoch = ContinuousQGANModelHandler.last_checkpoint_iteration(model_name)
    model = ContinuousQGANModelHandler()
    model.reload(model_name=model_name, epoch=epoch)

    results, images = compare(model, _ionq_runner(shots, seed), noise_models, n_samples, seed, smoothing)

    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(f"{out_dir}/fidelity_{model_name}_epoch={epoch}_shots={shots}.csv", index=False)
    plot_images(images, model.n_channels, f"{out_dir}/images_{model_name}_epoch={epoch}_shots={shots}.png")

    print(f"Fidelity of the generator output with the exact simulation ({model_name}, epoch {epoch}, {shots} shots):")
    print(results.groupby("noise_model", sort=False)["fidelity"].agg(["mean", "min", "max"]).to_string())
    print(f"Saved results to {out_dir}/")


if __name__ == "__main__":
    Fire(main)
