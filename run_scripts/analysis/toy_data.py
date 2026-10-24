import numpy as np


def make_bars_and_stripes(m=8, n_repeat=2, seed=0):
    """Classic small-circuit benchmark dataset: every possible all-bar and all-stripe
    pattern on an m x m grid, repeated and shuffled. Returns a flattened (N, m*m) array."""
    rng = np.random.default_rng(seed)
    patterns = []
    for bits in range(2 ** m):
        row = np.array([(bits >> i) & 1 for i in range(m)], dtype=float)
        patterns.append(np.tile(row, (m, 1)))
        patterns.append(np.tile(row, (m, 1)).T)
    patterns = np.unique(np.stack(patterns), axis=0)
    data = np.repeat(patterns, n_repeat, axis=0)
    rng.shuffle(data, axis=0)
    return data.reshape(len(data), -1)
