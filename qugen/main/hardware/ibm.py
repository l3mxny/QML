"""
Helpers to transpile and run generator circuits on IBM Quantum backends via Qiskit Runtime.

Backend names:
    'aer'           Ideal (noise-free) local simulation with finite shots.
    'fake_<name>'   Local simulation with the noise model and coupling map of a real IBM device snapshot,
                    e.g., 'fake_torino' (Heron) or 'fake_sherbrooke' (Eagle). Requires no IBM Quantum account.
    'least_busy'    The least busy real IBM Quantum device available to the saved account.
    otherwise       The real IBM Quantum device of this name, e.g., 'ibm_torino'.

Real devices require an IBM Quantum account saved beforehand via QiskitRuntimeService.save_account(...).
Credentials are never handled here.

Requires the optional dependencies: pip install ".[ibm]"
"""


def is_local_backend(backend_name):
    return backend_name == "aer" or backend_name.startswith("fake_")


def get_backend(backend_name, min_num_qubits=1):
    if backend_name == "aer":
        from qiskit_aer import AerSimulator
        return AerSimulator()
    if backend_name.startswith("fake_"):
        from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
        return FakeProviderForBackendV2().backend(backend_name)

    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()  # uses the saved account
    if backend_name == "least_busy":
        return service.least_busy(operational=True, simulator=False, min_num_qubits=min_num_qubits)
    return service.backend(backend_name)


def transpile(circuits, backend, optimization_level=3, seed_transpiler=0, initial_layout=None):
    """
    Transpile circuits into the backend's native gates and connectivity (instruction set architecture, ISA).

    All circuits are mapped onto the same physical qubits (chosen for the first circuit unless an initial layout is
    given), such that all generated images experience the same device noise.

    Returns:
        tuple[list[qiskit.QuantumCircuit], list[int]]: The ISA circuits and the physical qubit of each virtual qubit.
    """
    from qiskit.transpiler import generate_preset_pass_manager

    if initial_layout is None:
        pass_manager = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                                    seed_transpiler=seed_transpiler)
        first = pass_manager.run(circuits[0])
        initial_layout = first.layout.initial_index_layout(filter_ancillas=True)

    pass_manager = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend,
                                                seed_transpiler=seed_transpiler, initial_layout=list(initial_layout))
    return pass_manager.run(list(circuits)), list(initial_layout)


def two_qubit_gate_count(circuit):
    return sum(1 for instruction in circuit.data if instruction.operation.num_qubits == 2)


def resource_summary(circuit):
    """Gate counts and depths of a (transpiled) circuit, ignoring measurements and barriers."""
    ops = {k: v for k, v in circuit.count_ops().items() if k not in ("measure", "barrier")}
    return dict(
        n_two_qubit_gates=two_qubit_gate_count(circuit),
        n_gates=sum(ops.values()),
        depth=circuit.depth(lambda instruction: instruction.operation.name not in ("measure", "barrier")),
        two_qubit_depth=circuit.depth(lambda instruction: instruction.operation.num_qubits == 2),
        gate_counts=dict(sorted(ops.items())),
    )


def run_sampler(isa_circuits, backend, shots, dynamical_decoupling=True, twirling=False, job_tags=None):
    """
    Run the circuits as a single Sampler job (one primitive unified bloc per circuit).

    Error suppression options (dynamical decoupling, gate twirling) only apply to real devices.

    Returns:
        tuple[list[dict[str, int]], str | None]: Measurement counts per circuit and the job ID (None if local).
    """
    from qiskit_ibm_runtime import SamplerV2

    sampler = SamplerV2(mode=backend)
    local = is_local_backend(backend.name) or backend.name.startswith("aer")
    if not local:
        sampler.options.dynamical_decoupling.enable = dynamical_decoupling
        sampler.options.twirling.enable_gates = twirling
        if job_tags:
            sampler.options.environment.job_tags = list(job_tags)

    job = sampler.run(list(isa_circuits), shots=shots)
    job_id = None if local else job.job_id()
    if job_id is not None:
        print(f"Submitted job {job_id} to {backend.name}. Waiting for results...")
    result = job.result()
    counts = [pub_result.data.meas.get_counts() for pub_result in result]
    return counts, job_id
