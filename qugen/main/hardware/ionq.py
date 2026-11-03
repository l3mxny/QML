"""
Helpers to run generator circuits on IonQ backends (noisy cloud simulator or QPU) via qiskit-ionq.

Backend names:
    'ionq_simulator'  IonQ cloud simulator. Combine with a noise profile (see get_backend()) to simulate the noise of a
                      real IonQ device, e.g., 'aria-1'. The default profile 'ideal' is noise-free.
    'qpu.<name>'      The real IonQ device of this name, e.g., 'qpu.forte-1'.

Unlike on IBM Quantum, circuits are submitted via backend.run() and not via a Qiskit Sampler primitive, and they need no
local transpilation: the gates exported by qiskit_export (h, x, y, z, rx, ry, rz, cx, cz, swap, crx, cry, crz) are all
part of IonQ's QIS gate set, and the IonQ cloud compiles them into the native gates of the device.

Both backends need an IonQ API key, which qiskit-ionq reads from the environment variable IONQ_API_KEY (or a .env
file). Credentials are never handled here.

Requires the optional dependencies: pip install ".[ionq]"
"""

DEFAULT_NOISE_MODEL = "aria-1"


def get_backend(backend_name="ionq_simulator", noise_model=DEFAULT_NOISE_MODEL):
    """
    Get an IonQ backend.

    Args:
        backend_name (str, optional): 'ionq_simulator' or the name of a QPU such as 'qpu.forte-1'.
        noise_model (str, optional): Noise profile of the simulator, which is the name of the IonQ device whose noise is
            simulated, e.g., 'aria-1', or 'ideal' for no noise. Ignored for QPUs, which have their own noise.
    """
    from qiskit_ionq import IonQProvider

    backend = IonQProvider().get_backend(backend_name)
    if backend_name == "ionq_simulator":
        backend.set_options(noise_model=noise_model)
    return backend


def run_circuits(circuits, backend, shots, seed=None, job_tags=None):
    """
    Run the circuits as a single job and return their measurement counts.

    The IonQ simulator returns probabilities, which qiskit-ionq converts into counts by sampling on the client side.

    Args:
        circuits (list[qiskit.QuantumCircuit]): Circuits with measurements, e.g., from qiskit_export.generator_circuits().
        backend: The IonQ backend, see get_backend().
        shots (int): Number of shots per circuit.
        seed (int, optional): Seed of the sampling of the simulator. Ignored by QPUs.
        job_tags (list[str], optional): Tags to attach to the job.

    Returns:
        tuple[list[dict[str, int]], str]: Measurement counts per circuit and the job ID.
    """
    options = dict(shots=shots)
    if seed is not None:
        options["sampler_seed"] = seed
    if job_tags:
        options["extra_metadata"] = dict(tags=",".join(job_tags))  # job metadata holds strings only

    job = backend.run(list(circuits), **options)
    job_id = job.job_id()
    print(f"Submitted job {job_id} to {backend.name}. Waiting for results...")
    result = job.result()
    counts = [result.get_counts(i) for i in range(len(circuits))]
    return counts, job_id
