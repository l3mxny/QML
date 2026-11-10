import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler  # noqa: F401
from qugen.main.generator.measurements import decoder_factory

# Same import order as train_image_qgan.py: the model handler first, then x64
SAMPLE_COUNTS = """
import jax
import jax.numpy as jnp
import numpy as np

from qugen.main.generator.continuous_qgan_model_handler import ContinuousQGANModelHandler

jax.config.update("jax_enable_x64", True)
keys = jax.random.split(jax.random.PRNGKey(0), 20)
for n in (256, 2048):
    counts = np.stack([np.asarray(jax.random.multinomial(k, n=n, p=jnp.ones(32) / 32)) for k in keys])
    assert np.isfinite(counts).all() and (counts >= 0).all() and (counts.sum(axis=1) == n).all()
"""


def test_handler_does_not_enable_persistent_compilation_cache():
    assert not jax.config.jax_compilation_cache_dir


def test_multinomial_stays_valid_across_runs_in_one_folder(tmp_path):
    # A persistent compilation cache only misbehaves once programs are loaded from it, i.e., from the second run on.
    # By default JAX only caches programs that compile for over a second, which small ones miss on an idle machine.
    env = {**os.environ, "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0"}
    for _ in range(2):
        result = subprocess.run([sys.executable, "-c", SAMPLE_COUNTS], cwd=tmp_path, env=env, capture_output=True,
                                text=True)
        assert result.returncode == 0, result.stderr


def test_shots_decoding_gives_finite_images():
    probs = jnp.asarray(np.random.default_rng(0).dirichlet(np.ones(32), size=16))
    for scheme in ("FRQI", "FRQI_msb"):
        decoder = decoder_factory(scheme, n_ancilla_qubits=0, shots=512)
        assert np.isfinite(np.asarray(decoder(probs, rng_key=jax.random.PRNGKey(0)))).all()
