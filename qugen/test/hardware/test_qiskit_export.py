import jax
import numpy as np
import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_aer")

from qiskit.quantum_info import Statevector

from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler
from qugen.main.hardware import qiskit_export

# (generator_name, n_qubits, circuit_depth, decoding_scheme) for 4x4 images
GENERATORS = [
    # FRQI color rotation generators (paper ansatz incl. wire remapping of the color qubit), 4 address + 1 color qubit
    ("color_rot_skip_SO4_blocks_noise_angle_next_neighbor_mirror", 5, (2, 1), "FRQI_msb"),
    ("color_rot_skip_SO4_blocks_noise_angle_mirror", 5, (2, 1), "FRQI_msb"),
    # amplitude encoding generators, 4 address qubits
    ("SO4_blocks_noise_angle_mirror", 4, 2, "amplitude"),
    ("layered_rot_strongly_ent", 4, 2, "amplitude"),
]


def _random_model(generator_name, n_qubits, circuit_depth, decoding_scheme, seed=0):
    model = ContinuousQGANModelHandler()
    model.build(model_name="test", data_set="test", n_qubits=n_qubits, circuit_depth=circuit_depth,
                noise_distr="normal_multi_mode", noise_scale=0.1, noise_shift=list(range(10)), noise_tuning=10,
                transformation="minmax", measurement_scheme="comp_basis_probs", decoding_scheme=decoding_scheme,
                save_artifacts=False, discriminator_name=None, generator_name=generator_name)
    model.generator_weights = jax.random.normal(jax.random.PRNGKey(seed), shape=model.params_shape)
    return model


def _exact_measured_probs(circuit):
    """Exact probabilities of the measured qubits of a Qiskit circuit, ordered like counts_to_probs() does (classical
    bit 0 is the most significant bit)."""
    measured = {circuit.find_bit(instr.clbits[0]).index: circuit.find_bit(instr.qubits[0]).index
                for instr in circuit.data if instr.operation.name == "measure"}
    qubits = [measured[k] for k in sorted(measured)]
    circuit = circuit.remove_final_measurements(inplace=False)
    return Statevector(circuit).probabilities(qargs=list(reversed(qubits)))  # first qarg is least significant


@pytest.mark.parametrize("generator_name, n_qubits, circuit_depth, decoding_scheme", GENERATORS)
def test_exported_circuits_reproduce_generator_probs(generator_name, n_qubits, circuit_depth, decoding_scheme):
    model = _random_model(generator_name, n_qubits, circuit_depth, decoding_scheme)
    noise = qiskit_export.sample_noise(model, n_samples=4)

    circuits = qiskit_export.generator_circuits(model, noise)
    expected = qiskit_export.ideal_probs(model, noise)

    assert len(circuits) == 4
    assert all(c.num_qubits == n_qubits for c in circuits)
    exported = np.stack([_exact_measured_probs(c) for c in circuits])
    np.testing.assert_allclose(exported, expected, atol=1e-5)
    # the random weights must give distinct images, otherwise the comparison is trivial
    assert not np.allclose(expected[0], expected[1], atol=1e-3)


def test_counts_to_probs_bit_order():
    # Qiskit lists classical bit 0 last, PennyLane orders wire 0 as the most significant bit
    counts = {"001": 3, "100": 1}  # qubit 0 = 1 in the first bitstring, qubit 2 = 1 in the second
    probs = qiskit_export.counts_to_probs(counts, n_qubits=3)
    np.testing.assert_allclose(probs, [0, 0.25, 0, 0, 0.75, 0, 0, 0])

    smoothed = qiskit_export.counts_to_probs(counts, n_qubits=3, smoothing=1.)
    assert np.all(smoothed > 0) and np.isclose(smoothed.sum(), 1.)


def test_sampled_counts_match_generator_probs():
    from qugen.main.hardware import ibm

    model = _random_model(*GENERATORS[0])
    noise = qiskit_export.sample_noise(model, n_samples=2)
    circuits = qiskit_export.generator_circuits(model, noise)
    expected = qiskit_export.ideal_probs(model, noise)

    backend = ibm.get_backend("aer")
    isa_circuits, layout = ibm.transpile(circuits, backend)
    counts, job_id = ibm.run_sampler(isa_circuits, backend, shots=20000)
    estimated = np.stack([qiskit_export.counts_to_probs(c, n_qubits=model.n_qubits) for c in counts])

    assert job_id is None  # local simulation
    assert sorted(layout) == list(range(model.n_qubits))
    assert np.all(qiskit_export.classical_fidelity(estimated, expected) > 0.99)
