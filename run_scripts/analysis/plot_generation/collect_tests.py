"""Hardware, shot-noise and training-check data. Usage: python collect_tests.py <B|C|D|E> [...]"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from qiskit.quantum_info import Statevector

from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler
from qugen.main.generator.measurements import decoder_factory
from qugen.main.hardware import ibm, qiskit_export

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plotdata")
os.makedirs(OUT, exist_ok=True)
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LOGISTICS = os.path.join(REPO, "apps", "logistics")
DATA_PATH = os.path.join(LOGISTICS, "training_data", "mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area.npy")
TRAINED = {"K10": "continuous_mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area_minmax_qgan_a33dbb10",
           "K20": "continuous_mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area_minmax_qgan_6538ebe1"}
FRQI_NN = "color_rot_skip_SO4_blocks_noise_angle_next_neighbor_mirror"


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


def build(cls=ContinuousQGANModelHandler, generator_name=FRQI_NN, n_qubits=5, circuit_depth=(2, 1),
          decoding_scheme="FRQI_msb", save_artifacts=False, shots=None):
    return quiet(cls().build, model_name="t", data_set="t", n_qubits=n_qubits, circuit_depth=circuit_depth,
                 noise_distr="normal_multi_mode", noise_shift=list(range(10)), noise_tuning=10,
                 transformation="minmax", measurement_scheme="comp_basis_probs", decoding_scheme=decoding_scheme,
                 save_artifacts=save_artifacts, discriminator_name="convolutional", generator_name=generator_name,
                 gen_init_scale=0.05, gan_method="WGAN_10", shots=shots)


def exact_probs(circuit, qubit_order):
    """Exact probabilities with qubit_order[0] as the most significant bit."""
    sv = Statevector(circuit.remove_final_measurements(inplace=False))
    return sv.probabilities(qargs=list(reversed(qubit_order)))


def measured_order(circuit):
    measured = {circuit.find_bit(i.clbits[0]).index: circuit.find_bit(i.qubits[0]).index
                for i in circuit.data if i.operation.name == "measure"}
    return [measured[k] for k in sorted(measured)]


def reload_final(name):
    os.chdir(LOGISTICS)
    model = ContinuousQGANModelHandler()
    quiet(model.reload, model_name=name, epoch=10000)
    return model


def section_b():
    """Final checkpoints: exported circuits on Aer (ideal device) and fake_torino (device noise)."""
    out, info = {}, {}
    for key, name in TRAINED.items():
        model = reload_final(name)
        noise = qiskit_export.sample_noise(model, n_samples=20, seed=3)
        circuits = qiskit_export.generator_circuits(model, noise)
        probs = dict(ideal=qiskit_export.ideal_probs(model, noise),
                     before=np.stack([exact_probs(c, list(range(c.num_qubits))) for c in circuits]),
                     after=np.stack([exact_probs(c, measured_order(c)) for c in circuits]))
        info[key] = dict(model_name=name, epoch=10000, shots=4096, logical=ibm.resource_summary(circuits[0]))
        for backend_name in ["aer", "fake_torino"]:
            backend = ibm.get_backend(backend_name)
            t0 = time.time()
            isa, layout = ibm.transpile(circuits, backend)
            counts, _ = ibm.run_sampler(isa, backend, shots=4096)
            probs[backend_name] = np.stack([qiskit_export.counts_to_probs(c, model.n_qubits, smoothing=0.5)
                                            for c in counts])
            info[key][backend_name] = dict(layout=[int(q) for q in layout], seconds=time.time() - t0,
                                           **ibm.resource_summary(isa[0]))
            print(key, backend_name, f"{time.time() - t0:.0f}s", flush=True)
        for k, v in probs.items():
            out[f"{key}_p_{k}"] = v
            out[f"{key}_img_{k}"] = qiskit_export.decode_probs(model, v)
            out[f"{key}_fid_{k}"] = qiskit_export.classical_fidelity(v, probs["ideal"])
    np.savez(os.path.join(OUT, "B.npz"), **out)
    with open(os.path.join(OUT, "B.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)


def section_c():
    """Pixel error from shot noise alone: true multinomial sampling of the exact generator probabilities.

    NumPy's sampler, because jax.random.multinomial returns NaN and negative counts here. Sampled probabilities are
    treated like the decoder treats them (+1e-10, renormalized), then decoded without further noise.
    """
    out, rows = {}, []
    for key, name in TRAINED.items():
        model = reload_final(name)
        noise = qiskit_export.sample_noise(model, n_samples=500, seed=3)
        ideal = np.asarray(qiskit_export.ideal_probs(model, noise))
        ideal = ideal / ideal.sum(1, keepdims=True)
        exact = qiskit_export.decode_probs(model, ideal)
        out[f"{key}_shots_0"] = exact[:6]
        for shots in [128, 256, 512, 1024, 2048, 4096, 8192]:
            errors, rmses = [], []
            for seed in range(3):
                counts = np.random.default_rng(seed).multinomial(shots, ideal)
                sampled = counts / shots + 1e-10
                samples = qiskit_export.decode_probs(model, sampled / sampled.sum(1, keepdims=True))
                assert np.isfinite(samples).all()
                errors.append(np.abs(samples - exact).mean())
                rmses.append(np.sqrt(((samples - exact) ** 2).mean()))
            rows.append(dict(model=key, shots=shots, pixel_mae=float(np.mean(errors)), pixel_rmse=float(np.mean(rmses))))
            if key == "K10":
                out[f"{key}_shots_{shots}"] = samples[:6]
    np.savez(os.path.join(OUT, "C.npz"), **out)
    with open(os.path.join(OUT, "C.json"), "w") as f:
        json.dump(rows, f, indent=1)


def train_and_log(name, n_epochs, cls=ContinuousQGANModelHandler, lr=(2.5e-3, 2.5e-4), **build_kwargs):
    data = np.load(DATA_PATH)[:8192] / 255.
    np.random.seed(0)
    model = build(cls=cls, save_artifacts=True, **build_kwargs)
    t0 = time.time()
    quiet(model.train, data, n_epochs=n_epochs, initial_learning_rate_generator=lr[0],
          initial_learning_rate_discriminator=lr[1], adam_b1=0.5, adam_b2=0.9, batch_size=64,
          discriminator_training_steps=10)
    seconds = time.time() - t0
    log = np.genfromtxt(os.path.join(model.path_to_models, "evaluation_summary.csv"), delimiter=",", names=True)
    bad = np.where(~(np.isfinite(log["loss_generator"]) & np.isfinite(log["loss_discriminator"])))[0]
    print(f"{name}: {seconds:.0f}s, first NaN iteration {int(log['iteration'][bad[0]]) if len(bad) else None}", flush=True)
    return dict(name=name, seconds=seconds, first_nan=int(log["iteration"][bad[0]]) if len(bad) else None,
                iteration=log["iteration"], loss_generator=log["loss_generator"],
                loss_discriminator=log["loss_discriminator"], grad_generator=log["grad_mag_generator"],
                grad_discriminator=log["grad_mag_discriminator"])


def save_runs(runs, section):
    np.savez(os.path.join(OUT, f"{section}.npz"), **{f"{i}__{k}": v for i, r in enumerate(runs) for k, v in r.items()
                                                     if isinstance(v, np.ndarray)})
    with open(os.path.join(OUT, f"{section}.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if not isinstance(v, np.ndarray)} for r in runs], f, indent=2)


def section_d():
    """Short trainings on the current code. NaN checks are off so a failing run still logs where it failed."""
    os.chdir(HERE)  # model folders go to ./experiments next to this script (git-ignored)
    runs = [train_and_log("FRQI, exact", 150),
            train_and_log("FRQI, 256 shots", 150, shots=256),
            train_and_log("FRQI, 2048 shots", 150, shots=2048),
            train_and_log("Amplitude, exact", 150, lr=(1e-2, 1e-3), generator_name="SO4_blocks_noise_angle_mirror",
                          n_qubits=4, circuit_depth=4, decoding_scheme="amplitude")]
    save_runs(runs, "D")


def section_e():
    """Same seed with the training step uncompiled (rebuilt from the current source) and compiled."""
    os.chdir(HERE)
    with open(os.path.join(REPO, "qugen", "main", "generator", "continuous_qgan_model_handler.py"), encoding="utf-8") as f:
        source = f.read()
    for old, new in [("jax.jit(jax.value_and_grad(cost_fn_discriminator, argnums=3))",
                      "jax.value_and_grad(cost_fn_discriminator, argnums=3)"),
                     ("jax.jit(jax.value_and_grad(cost_fn_generator, argnums=1))",
                      "jax.value_and_grad(cost_fn_generator, argnums=1)"),
                     ("return jax.jit(update_fn)", "return update_fn")]:
        assert old in source, old
        source = source.replace(old, new)
    path = os.path.join(HERE, "handler_eager.py")
    with open(path, "w", encoding="utf-8") as f:
        f.write(source)
    spec = importlib.util.spec_from_file_location("handler_eager", path)
    handler_eager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(handler_eager)
    save_runs([train_and_log("Uncompiled", 40, cls=handler_eager.ContinuousQGANModelHandler),
               train_and_log("Compiled", 40)], "E")


if __name__ == "__main__":
    if set(sys.argv[1:]) & set("CDE"):  # match train_image_qgan.main; must be set before any JAX work in this process
        jax.config.update("jax_enable_x64", True)
    for section in sys.argv[1:]:
        t0 = time.time()
        globals()[f"section_{section.lower()}"]()
        print(f"section {section} done in {time.time() - t0:.0f}s", flush=True)
