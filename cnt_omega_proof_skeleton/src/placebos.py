"""
Placebo/null generators: time-shuffle, phase randomize, label shift, sign flip.
(Implement locally against your dataset loaders.)
"""
def time_shuffle(x, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    idx = np.arange(len(x))
    rng.shuffle(idx)
    return x[idx]
