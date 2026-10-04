# Phase 1: train a small grid of circuit configs (generator architecture x qubit count x
# circuit depth) for a short, fixed iteration budget, and log KL divergence + trainable
# parameter count for each. Meant to run on Colab/Kaggle, not locally.
#
# Example:
#   python run_scripts/analysis/small_circuit_sweep.py \
#       --generator_names='("layered_rot_strongly_ent","SO4_blocks")' --qubit_counts='(5,7)'

import itertools
import os

import fire
import pandas as pd

from qugen.main.data.metrics_factory import metrics_factory
from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler
from toy_data import make_bars_and_stripes


def run_one(generator_name, n_qubits, circuit_depth, data, n_epochs, batch_size,
            lr_gen, lr_disc, discriminator_training_steps, shots, metric_fn):
    model = ContinuousQGANModelHandler()
    model.build(
        model_name="sweep",
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
    model.train(
        data,
        n_epochs=n_epochs,
        initial_learning_rate_generator=lr_gen,
        initial_learning_rate_discriminator=lr_disc,
        batch_size=batch_size,
        discriminator_training_steps=discriminator_training_steps,
    )
    synthetic = model.sample(n_samples=len(data))
    kl = metric_fn(data, synthetic)
    return float(kl), int(model.num_params)


def main(
    generator_names=("layered_rot_strongly_ent", "SO4_blocks", "SO4_blocks_mirror", "SO4_full_blocks"),
    qubit_counts=(5, 7),
    circuit_depths=(1, 2, 3),
    n_epochs=150,
    batch_size=32,
    lr_gen=0.005,
    lr_disc=0.001,
    discriminator_training_steps=1,
    shots=None,
    out_csv="run_scripts/analysis/results/phase1_small_circuit_sweep.csv",
):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    data = make_bars_and_stripes()
    metric_fn = metrics_factory("kl_div")

    rows = []
    for gen_name, n_qubits, depth in itertools.product(generator_names, qubit_counts, circuit_depths):
        print(f"=== {gen_name} | n_qubits={n_qubits} | depth={depth} ===")
        try:
            kl, n_params = run_one(gen_name, n_qubits, depth, data, n_epochs, batch_size,
                                    lr_gen, lr_disc, discriminator_training_steps, shots, metric_fn)
            status = "ok"
        except Exception as e:
            kl, n_params = float("nan"), None
            status = f"error: {e}"
        rows.append(dict(generator_name=gen_name, n_qubits=n_qubits, circuit_depth=depth,
                          n_params=n_params, kl_div=kl, status=status))
        # Write after every run so a Colab/Kaggle disconnect doesn't lose completed progress.
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"Saved {len(rows)} runs to {out_csv}")


if __name__ == "__main__":
    fire.Fire(main)
