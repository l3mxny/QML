"""
Export the generator circuits of trained QGANs to Qiskit, e.g., to run them on IBM Quantum hardware.

On hardware, the generator is only used for inference: every noise sample defines one circuit (the noise is uploaded
as rotation angles), whose computational basis probabilities are estimated from measurement counts and then decoded
into an image in exactly the same way as the simulated probabilities during training.

Bit ordering: PennyLane orders computational basis states with wire 0 as the most significant bit. Qiskit count
bitstrings list classical bit 0 last. The exported circuits measure qubit i into classical bit i, and
counts_to_probs() converts the counts back into PennyLane's ordering.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pennylane as qml

from qugen.main.generator.measurements import decoder_factory

# PennyLane operation name -> qiskit.QuantumCircuit method name (all other operations are decomposed into these)
QISKIT_GATES = {
    "Hadamard": "h",
    "PauliX": "x",
    "PauliY": "y",
    "PauliZ": "z",
    "RX": "rx",
    "RY": "ry",
    "RZ": "rz",
    "CNOT": "cx",
    "CZ": "cz",
    "SWAP": "swap",
    "CRX": "crx",
    "CRY": "cry",
    "CRZ": "crz",
}


def unjit(fn):
    """Return the function (e.g., the generator QNode) wrapped by jax.jit."""
    return getattr(fn, "__wrapped__", fn)


def sample_noise(model, n_samples, seed=0):
    """Sample generator noise inputs in the same way as ContinuousQGANModelHandler.predict_transform()."""
    key_noise, key_init_noise = jax.random.split(jax.random.PRNGKey(seed))
    noise = model.noise_sample_fn(key=key_noise, shape=(n_samples, *model.noise_shape),
                                  scale=model.noise_scale, shift=model.noise_shift)
    noise = noise.reshape(n_samples, *model.noise_shape)  # multi-mode noise is squeezed for a single sample
    if model.init_noise_distr is not None:
        init_noise = model.init_noise_sample_fn(key=key_init_noise, shape=(n_samples, *model.init_noise_shape),
                                                scale=model.init_noise_scale, shift=model.init_noise_shift)
        noise = (noise, init_noise.reshape(n_samples, *model.init_noise_shape))
    return noise


def _iter_noise(noise):
    if isinstance(noise, tuple):  # (noise, initial noise) pairs
        return zip(*noise)
    return iter(noise)


def generator_tapes(model, noise):
    """
    Construct one PennyLane tape per noise sample for the generator of a built (or reloaded) model handler.

    Args:
        model (ContinuousQGANModelHandler): Model handler with generator weights.
        noise (array or tuple of arrays): Noise inputs of shape (n_samples, *model.noise_shape), see sample_noise().

    Returns:
        list[qml.tape.QuantumScript]: The generator circuit for each noise sample (incl. all user transforms such as
        the wire mapping of the FRQI generators, but before any device-specific compilation).
    """
    qnode = unjit(model.generator)
    weights = jnp.asarray(model.generator_weights)
    return [qml.workflow.construct_tape(qnode, level="user")(z, weights) for z in _iter_noise(noise)]


def tape_to_qiskit(tape, n_qubits=None, measure=True):
    """
    Convert a PennyLane tape with integer wires 0, ..., n_qubits - 1 into a qiskit.QuantumCircuit.

    Args:
        tape (qml.tape.QuantumScript): The circuit to convert. Its measurements are ignored.
        n_qubits (int, optional): Number of qubits of the Qiskit circuit. Defaults to the number of tape wires.
        measure (bool, optional): Whether to measure all qubits (qubit i into classical bit i). Defaults to True.

    Returns:
        qiskit.QuantumCircuit: The equivalent Qiskit circuit.
    """
    from qiskit import QuantumCircuit

    operations = [op for op in tape.operations if not isinstance(op, qml.Barrier)]
    [tape], _ = qml.transforms.decompose(tape.copy(operations=operations), gate_set=set(QISKIT_GATES))

    if n_qubits is None:
        n_qubits = len(tape.wires)
    circuit = QuantumCircuit(n_qubits)
    for op in tape.operations:
        params = [float(np.real(np.asarray(p))) for p in op.parameters]
        qubits = [int(w) for w in op.wires]
        assert all(0 <= q < n_qubits for q in qubits), f"Wires {op.wires} of {op} exceed {n_qubits} qubits"
        getattr(circuit, QISKIT_GATES[op.name])(*params, *qubits)

    if measure:
        circuit.measure_all()
    return circuit


def generator_circuits(model, noise):
    """Qiskit circuits (with measurements) of the generator for each noise sample, see generator_tapes()."""
    n_qubits = model.n_qubits + model.n_ancilla_qubits
    return [tape_to_qiskit(tape, n_qubits=n_qubits) for tape in generator_tapes(model, noise)]


def counts_to_probs(counts, n_qubits, smoothing=0.):
    """
    Convert Qiskit measurement counts into a probability vector in PennyLane ordering (wire 0 is the most significant
    bit), i.e., the format of the comp_basis_probs measurement scheme.

    Args:
        counts (dict[str, int]): Counts of measured bitstrings, where the last character is classical bit 0.
        n_qubits (int): Number of measured qubits.
        smoothing (float, optional): Additive (Laplace) smoothing pseudo-count per basis state. A positive value
            avoids zero probabilities for unobserved basis states, which the FRQI decoding cannot handle (a pixel
            whose two basis states are both unobserved has no defined brightness). Defaults to 0.

    Returns:
        np.ndarray: Estimated probabilities of shape (2 ** n_qubits,).
    """
    probs = np.full(2 ** n_qubits, float(smoothing))
    for bitstring, count in counts.items():
        bitstring = bitstring.replace(" ", "")
        assert len(bitstring) == n_qubits, f"Bitstring {bitstring} does not match {n_qubits} qubits"
        probs[int(bitstring[::-1], 2)] += count
    return probs / probs.sum()


def ideal_probs(model, noise):
    """Exact (simulated, shot-free) generator output probabilities for the given noise inputs."""
    v_qnode = jax.vmap(lambda z: model.generator(z, model.generator_weights))
    return np.asarray(model.standardize_pennylane_output(v_qnode(noise)))


def decode_probs(model, probs):
    """
    Decode generator output probabilities into images in the original data space, like model.predict() does for
    simulated outputs, but without adding simulated shot noise (estimated probabilities already include it).

    Args:
        model (ContinuousQGANModelHandler): The (reloaded) model handler.
        probs (np.ndarray): Probabilities of shape (n_samples, 2 ** n_qubits) in PennyLane ordering.

    Returns:
        np.ndarray: Images of shape (n_samples, n_pixels * n_channels).
    """
    decoder = decoder_factory(model.decoding_scheme, n_ancilla_qubits=model.n_ancilla_qubits, shots=None)
    samples = np.asarray(decoder(jnp.asarray(probs)))
    return np.asarray(model.normalizer.inverse_transform(samples))


def classical_fidelity(p, q):
    """Classical (Bhattacharyya) fidelity (sum_i sqrt(p_i q_i))^2 between rows of two probability arrays."""
    return np.sum(np.sqrt(np.clip(p, 0, None) * np.clip(q, 0, None)), axis=-1) ** 2
