"""Draw the full circuit of every tested configuration and record its resources.

Usage: python draw_circuits.py [config ids]   (default: all)
Writes plots_small_circuits/circuits/<id>_<name>.{png,pdf} and plotdata/circuits/<id>.json + <id>_map.npy.
Depths come from the exported circuit itself, not from the drawing (barriers only separate layers visually).
"""
import contextlib
import io
import json
import os
import sys
import time

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pennylane as qml  # noqa: E402
from qiskit import ClassicalRegister, QuantumCircuit  # noqa: E402
from qiskit.converters import circuit_to_dag  # noqa: E402

from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler  # noqa: E402
from qugen.main.hardware import ibm, qiskit_export  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
LOGISTICS = os.path.join(REPO, "apps", "logistics")
OUT = os.path.join(REPO, "plots_small_circuits", "circuits")
DATA = os.path.join(HERE, "plotdata", "circuits")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)

K10 = "continuous_mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area_minmax_qgan_a33dbb10"
K20 = "continuous_mnist_0_1_2_3_4_5_6_7_8_9_4x4_N_60000_area_minmax_qgan_6538ebe1"
FRQI_NN = "color_rot_skip_SO4_blocks_noise_angle_next_neighbor_mirror"
FRQI = "color_rot_skip_SO4_blocks_noise_angle_mirror"
AMP_SO4 = "SO4_blocks_noise_angle_mirror"
AMP_SEL = "layered_rot_strongly_ent"
FRQI_ROLES = ("q0 = color qubit (target of every CRY), q1-q4 = address qubits. Measured in order q1 q2 q3 q4 q0. "
              "RX = noise input, RY and CRY = trained weights.")
AMP_ROLES = "All qubits are address qubits. Measured in order q0 q1 q2 q3. RX = noise input, other rotations = trained weights."

CONFIGS = {
    "c01": dict(name="frqi_next_neighbor_2_1", title="FRQI color rotation, next-neighbor blocks, depth (2,1), 5 qubits",
                generator=FRQI_NN, n_qubits=5, depth=(2, 1), decoding="FRQI_msb", roles=FRQI_ROLES,
                used="test_qiskit_export (exact + 20k-shot Aer), training checks (exact, 256, 2048 shots), compiled-step check"),
    "c02": dict(name="frqi_so4_2_1", title="FRQI color rotation, SO(4) blocks, depth (2,1), 5 qubits",
                generator=FRQI, n_qubits=5, depth=(2, 1), decoding="FRQI_msb", roles=FRQI_ROLES,
                used="test_qiskit_export"),
    "c03": dict(name="amplitude_so4_2", title="Amplitude, SO(4) blocks, depth 2, 4 qubits",
                generator=AMP_SO4, n_qubits=4, depth=2, decoding="amplitude", roles=AMP_ROLES,
                used="test_qiskit_export"),
    "c04": dict(name="amplitude_strongly_entangling_2", title="Amplitude, strongly entangling layers, depth 2, 4 qubits",
                generator=AMP_SEL, n_qubits=4, depth=2, decoding="amplitude", roles=AMP_ROLES,
                used="test_qiskit_export"),
    "c05": dict(name="amplitude_so4_4", title="Amplitude, SO(4) blocks, depth 4, 4 qubits",
                generator=AMP_SO4, n_qubits=4, depth=4, decoding="amplitude", roles=AMP_ROLES,
                used="training check (amplitude, exact)"),
    "c06": dict(name="baseline_K10_8_2", title="Baseline K = 10, depth (8,2), 5 qubits, trained weights (iteration 10,000)",
                trained=K10, roles=FRQI_ROLES, used="4x4 baseline training, Aer and fake_torino export runs"),
    "c07": dict(name="baseline_K20_8_2", title="Baseline K = 20, depth (8,2), 5 qubits, trained weights (iteration 10,000)",
                trained=K20, roles=FRQI_ROLES, used="4x4 baseline training, Aer and fake_torino export runs"),
    "c08": dict(name="c01_compiled_aer", title="Configuration c01 compiled for Aer", base="c01", backend="aer",
                roles="Compiled at optimization level 3 for a simulator with no wiring limits; qubits keep their numbering.",
                used="test_sampled_counts_match_generator_probs"),
    "c09": dict(name="baseline_K10_compiled_aer", title="Baseline K = 10 compiled for Aer", base="c06", backend="aer",
                roles="Compiled at optimization level 3 for a simulator with no wiring limits; qubits keep their numbering.",
                used="Aer export run (4096 shots)"),
    "c10": dict(name="baseline_K10_compiled_fake_torino", title="Baseline K = 10 compiled for fake_torino (CZ, RZ, SX, X)",
                base="c06", backend="fake_torino", roles="Physical qubits of the device; routing swaps are expanded into CZ gates.",
                used="fake_torino export run (4096 shots)"),
    "c11": dict(name="baseline_K20_compiled_fake_torino", title="Baseline K = 20 compiled for fake_torino (CZ, RZ, SX, X)",
                base="c07", backend="fake_torino", roles="Physical qubits of the device; routing swaps are expanded into CZ gates.",
                used="fake_torino export run (4096 shots)"),
}

BLUE, ORANGE, AQUA, INK, MUTED, SURFACE = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#898781", "#fcfcfb"
STYLE = dict(name="bw", backgroundcolor=SURFACE, fontsize=11, subfontsize=8, linecolor="#52514e", textcolor=INK,
             gatetextcolor=INK, subtextcolor=INK, creglinecolor=MUTED, barrierfacecolor="#e1e0d9",
             displaycolor={"h": ["#c3c2b7", INK], "rx": [AQUA, INK], "ry": [BLUE, "#ffffff"], "rz": ["#cde2fb", INK],
                           "cx": [INK, "#ffffff"], "cry": [ORANGE, INK], "cz": [INK, "#ffffff"], "sx": ["#9ec5f4", INK],
                           "x": ["#c3c2b7", INK], "u1": ["#cde2fb", INK], "u2": ["#9ec5f4", INK],
                           "u3": ["#86b6ef", INK], "unitary": ["#e8a7bd", INK], "rxx": [ORANGE, INK],
                           "rzz": [ORANGE, INK], "cp": [ORANGE, INK], "meas": ["#f0efec", INK]})


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*args, **kwargs)


def get_model(cfg):
    if "trained" in cfg:
        os.chdir(LOGISTICS)
        model = ContinuousQGANModelHandler()
        quiet(model.reload, model_name=cfg["trained"], epoch=10000)
        return model
    model = quiet(ContinuousQGANModelHandler().build, model_name="t", data_set="t", n_qubits=cfg["n_qubits"],
                  circuit_depth=cfg["depth"], noise_distr="normal_multi_mode", noise_scale=0.1,
                  noise_shift=list(range(10)), noise_tuning=10, transformation="minmax",
                  measurement_scheme="comp_basis_probs", decoding_scheme=cfg["decoding"], save_artifacts=False,
                  discriminator_name=None, generator_name=cfg["generator"])
    model.generator_weights = jax.random.normal(jax.random.PRNGKey(0), shape=model.params_shape)  # as in the tests
    return model


def with_barriers(tape, n_qubits):
    """The exported circuit with the generator's own layer boundaries kept as barriers, for drawing only."""
    segments, current = [], []
    for op in tape.operations:
        if isinstance(op, qml.Barrier):
            if current:
                segments.append(current)
            current = []
        else:
            current.append(op)
    if current:
        segments.append(current)
    circuit = QuantumCircuit(n_qubits)
    for i, ops in enumerate(segments):
        circuit.compose(qiskit_export.tape_to_qiskit(tape.copy(operations=ops), n_qubits=n_qubits, measure=False),
                        inplace=True)
        if i < len(segments) - 1:
            circuit.barrier()
    circuit.add_register(ClassicalRegister(n_qubits, name="meas"))
    circuit.measure([int(w) for w in tape.measurements[0].wires], list(range(n_qubits)))
    return circuit


def gate_map(circuit):
    """Wires x moments matrix of a circuit: 0 idle, 1 single-qubit gate, 2 two-qubit gate. Width = circuit depth."""
    bare = circuit.remove_final_measurements(inplace=False)
    used = sorted({bare.find_bit(q).index for i in bare.data for q in i.qubits})
    row = {q: r for r, q in enumerate(used)}
    layers = list(circuit_to_dag(bare).layers())
    m = np.zeros((len(used), len(layers)), dtype=np.int8)
    for t, layer in enumerate(layers):
        for node in layer["graph"].op_nodes():
            for q in node.qargs:
                m[row[bare.find_bit(q).index], t] = 2 if len(node.qargs) > 1 else 1
    return m


def draw(circuit, cfg, path, fold, summary):
    t0 = time.time()
    fig = circuit.draw("mpl", fold=fold, idle_wires=False, style=STYLE, cregbundle=True, scale=0.7)
    header = (f"{cfg['title']}\n{summary['n_gates']} gates, {summary['n_two_qubit_gates']} two-qubit gates, "
              f"depth {summary['depth']}, two-qubit depth {summary['two_qubit_depth']}. {cfg['roles']}\n"
              f"Used in: {cfg['used']}.")
    fig.text(0.01, 1.0, header, va="bottom", ha="left", fontsize=max(10, fig.get_figwidth() * 0.42), color=INK)
    fig.savefig(path + ".pdf", bbox_inches="tight")
    size = fig.get_size_inches()
    dpi = min(110, 9000 / max(size))  # stays readable when zoomed, without exceeding what image viewers accept
    fig.savefig(path + ".png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"  drawn in {time.time() - t0:.0f}s, {size[0]:.0f}x{size[1]:.0f} in at {dpi:.0f} dpi", flush=True)


cache, backends = {}, {}


def exported(cid):
    """(exact exported circuit, tape, model) of a configuration, cached so compiled variants use the same circuit."""
    if cid not in cache:
        cfg = CONFIGS[cid]
        model = get_model(cfg)
        # trained baselines: first circuit of the exact 20-sample export run; tests: a noise sample as in the tests
        noise = qiskit_export.sample_noise(model, n_samples=20 if "trained" in cfg else 4, seed=3 if "trained" in cfg else 0)
        tape = quiet(qiskit_export.generator_tapes, model, jax.tree.map(lambda x: x[:1], noise))[0]
        cache[cid] = (qiskit_export.tape_to_qiskit(tape, n_qubits=model.n_qubits + model.n_ancilla_qubits), tape, model)
    return cache[cid]


def compiled(circuit, backend_name):
    backends.setdefault(backend_name, ibm.get_backend(backend_name))
    return ibm.transpile([circuit], backends[backend_name])


for cid in (sys.argv[1:] or CONFIGS):
    cfg = CONFIGS[cid]
    t0 = time.time()
    if "base" in cfg:
        circuit, layout = compiled(exported(cfg["base"])[0], cfg["backend"])
        circuit, drawn = circuit[0], circuit[0]
        n_params, generator = None, cfg["base"]
    else:
        circuit, tape, model = exported(cid)
        drawn = with_barriers(tape, circuit.num_qubits)
        layout = None
        n_params, generator = int(model.num_params), model.generator_name
        n_ops = {k: v for k, v in drawn.count_ops().items() if k != "barrier"}
        assert n_ops == dict(circuit.count_ops()), (n_ops, dict(circuit.count_ops()))  # drawing == exported circuit
    summary = ibm.resource_summary(circuit)
    gates = gate_map(circuit)
    assert gates.shape[1] == summary["depth"], (gates.shape, summary["depth"])
    record = dict(id=cid, title=cfg["title"], used=cfg["used"], generator=generator, n_params=n_params,
                  n_qubits=int(gates.shape[0]), layout=[int(q) for q in layout] if layout else None, **summary)
    if "base" not in cfg:  # what this circuit costs on the device
        isa, device_layout = compiled(circuit, "fake_torino")
        record["torino"] = ibm.resource_summary(isa[0])
    np.save(os.path.join(DATA, f"{cid}_map.npy"), gates)
    with open(os.path.join(DATA, f"{cid}.json"), "w") as f:
        json.dump(record, f, indent=1)
    print(cid, cfg["name"], {k: record[k] for k in ["n_gates", "n_two_qubit_gates", "depth", "two_qubit_depth"]},
          record.get("torino", {}).get("n_two_qubit_gates"), flush=True)
    # small circuits on one row (fold=-1), large ones folded into rows of 120 columns
    draw(drawn, cfg, os.path.join(OUT, f"{cid}_{cfg['name']}"), fold=120 if summary["n_gates"] > 200 else -1, summary=summary)
