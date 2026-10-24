# Phase 2: sweep `shots` and measure its effect on (a) an already-trained generator's
# decoded output quality, and (b) training dynamics when trained from scratch at that
# shot count. Requires the Phase 0 numerical-safety fix in decodings.py (clipping
# probabilities away from exact zero before jax.random.multinomial) to run reliably
# across many shot values unattended -- without it, some configs will hit NaN.
#
# Example:
#   python run_scripts/analysis/shot_count_sweep.py --n_qubits=7 --circuit_depth=2

import os

import fire
import numpy as np
import pandas as pd

from qugen.main.data.metrics_factory import metrics_factory
from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler
from qugen.main.generator.measurements import decoder_factory
from toy_data import make_bars_and_stripes

DEFAULT_SHOTS = (None, 64, 128, 256, 512, 1024, 2048, 4096)


def shots_label(shots):
    # "None" (Python's own str(None)) collides with pandas' default missing-value tokens
    # and silently round-trips to NaN on CSV read -- use an unambiguous label instead.
    return "exact" if shots is None else str(shots)


def build_model(n_qubits, circuit_depth, generator_name, shots):
    model = ContinuousQGANModelHandler()
    model.build(
        model_name="shots_sweep",
        data_set="bars_and_stripes",
        n_qubits=n_qubits,
        circuit_depth=circuit_depth,
        noise_distr="normal",
        noise_scale=0.3,
        transformation="minmax",
        measurement_scheme="comp_basis_probs",
        decoding_scheme="FRQI",
        discriminator_name="continuous_fully_connected",
        generator_name=generator_name,
        gen_init_scale=0.5,
        shots=shots,
        save_artifacts=False,
    )
    return model


def decode_only_sweep(data, n_qubits, circuit_depth, generator_name, shots_values, n_epochs, metric_fn):
    """Trains once, then only swaps the decoder's `shots` setting (same pattern the
    model handler's own reload() uses) -- isolates measurement-noise effects from
    training dynamics, since the generator's weights never change across this loop."""
    model = build_model(n_qubits, circuit_depth, generator_name, shots=None)
    model.train(data, n_epochs=n_epochs, initial_learning_rate_generator=0.005,
                initial_learning_rate_discriminator=0.001, batch_size=32,
                discriminator_training_steps=1)

    rows = []
    for shots in shots_values:
        model.shots = shots
        model.decoder = decoder_factory(model.decoding_scheme, n_ancilla_qubits=model.n_ancilla_qubits, shots=shots)
        synthetic = model.sample(n_samples=len(data))
        kl = metric_fn(data, synthetic)
        rows.append(dict(mode="decode_only", shots=shots_label(shots), kl_div=float(kl)))
    return rows


def training_sensitivity_sweep(data, n_qubits, circuit_depth, generator_name, shots_values, n_epochs, metric_fn):
    """Trains a fresh model at each shot count for the same short, fixed budget."""
    rows = []
    for shots in shots_values:
        model = build_model(n_qubits, circuit_depth, generator_name, shots=shots)
        model.train(data, n_epochs=n_epochs, initial_learning_rate_generator=0.005,
                    initial_learning_rate_discriminator=0.001, batch_size=32,
                    discriminator_training_steps=1)
        synthetic = model.sample(n_samples=len(data))
        kl = metric_fn(data, synthetic)
        rows.append(dict(mode="train_from_scratch", shots=shots_label(shots), kl_div=float(kl)))
    return rows


def main(
    n_qubits=7,
    circuit_depth=2,
    generator_name="layered_rot_strongly_ent",
    shots_values=DEFAULT_SHOTS,
    n_epochs=150,
    out_csv="run_scripts/analysis/results/phase2_shot_count_sweep.csv",
):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    data = make_bars_and_stripes()
    metric_fn = metrics_factory("kl_div")

    rows = []
    rows += decode_only_sweep(data, n_qubits, circuit_depth, generator_name, shots_values, n_epochs, metric_fn)
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    rows += training_sensitivity_sweep(data, n_qubits, circuit_depth, generator_name, shots_values, n_epochs, metric_fn)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(rows)} rows to {out_csv}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for mode, group in df.groupby("mode"):
            x = [1e6 if s == "exact" else int(s) for s in group["shots"]]  # "exact" (no shot noise) plotted far right
            order = np.argsort(x)
            ax.plot(np.array(x)[order], group["kl_div"].to_numpy()[order], marker="o", label=mode)
        ax.set_xscale("log")
        ax.set_xlabel("shots (log scale; rightmost point = exact, no shot noise)")
        ax.set_ylabel("KL divergence (lower = better)")
        ax.legend()
        ax.set_title(f"Shot count vs. quality ({generator_name}, {n_qubits}q, depth={circuit_depth})")
        plot_path = out_csv.replace(".csv", ".png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {plot_path}")
    except Exception as e:
        print(f"Plotting skipped: {e}")


if __name__ == "__main__":
    fire.Fire(main)
