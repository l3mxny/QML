import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_ionq")

import qiskit_ionq
from qiskit import QuantumCircuit
from qiskit_ionq.helpers import ionq_basis_gates

from qugen.main.hardware import ionq


class FakeResult:
    def __init__(self, counts):
        self._counts = counts

    def get_counts(self, experiment):
        return self._counts[experiment]


class FakeJob:
    def __init__(self, counts):
        self._counts = counts

    def job_id(self):
        return "job-123"

    def result(self):
        return FakeResult(self._counts)


class FakeBackend:
    """Stands in for an IonQ backend, so the tests need neither an API key nor network access."""
    name = "ionq_simulator"

    def __init__(self, counts=None):
        self.options = {}
        self.run_calls = []
        self._counts = counts

    def set_options(self, **options):
        self.options.update(options)

    def run(self, circuits, **options):
        self.run_calls.append((circuits, options))
        return FakeJob(self._counts)


def _circuit():
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit


@pytest.fixture
def fake_provider(monkeypatch):
    requested = []

    class FakeProvider:
        def get_backend(self, name):
            requested.append(name)
            return FakeBackend()

    monkeypatch.setattr(qiskit_ionq, "IonQProvider", FakeProvider)
    return requested


def test_simulator_backend_gets_noise_profile(fake_provider):
    backend = ionq.get_backend("ionq_simulator", noise_model="aria-1")
    assert fake_provider == ["ionq_simulator"]
    assert backend.options == dict(noise_model="aria-1")


def test_default_noise_profile_is_aria_1(fake_provider):
    assert ionq.get_backend().options == dict(noise_model="aria-1")


def test_qpu_backend_gets_no_noise_profile(fake_provider):
    backend = ionq.get_backend("qpu.forte-1")
    assert fake_provider == ["qpu.forte-1"]
    assert backend.options == {}


def test_run_circuits_returns_counts_per_circuit_in_order():
    counts = [{"00": 60, "11": 40}, {"01": 100}]
    backend = FakeBackend(counts)
    circuits = [_circuit(), _circuit()]

    result, job_id = ionq.run_circuits(circuits, backend, shots=100)

    assert result == counts
    assert job_id == "job-123"
    assert len(backend.run_calls) == 1  # one job for all circuits
    submitted, options = backend.run_calls[0]
    assert submitted == circuits
    assert options == dict(shots=100)


def test_run_circuits_passes_seed_and_tags():
    backend = FakeBackend([{"00": 10}])
    ionq.run_circuits([_circuit()], backend, shots=10, seed=7, job_tags=["qgan", "noisy"])
    _, options = backend.run_calls[0]
    assert options == dict(shots=10, sampler_seed=7, extra_metadata=dict(tags="qgan,noisy"))


def test_exported_gates_are_supported_by_ionq():
    pytest.importorskip("pennylane")
    from qugen.main.hardware.qiskit_export import QISKIT_GATES

    # all gates are submitted as they are, i.e., without local transpilation
    assert set(QISKIT_GATES.values()) <= set(ionq_basis_gates)
