import numpy as np
import minfs.utils as utils


def single_minfs(X, y):
    X = X.astype(np.uint8)
    y = y.astype(np.uint8)

    if np.all(y) or not np.any(y):
        # constant target - no solutions
        return []

    C, Np = utils.packed_coverage_map(X, y)
    fs = set()
    utils.greedy_repair(C, Np, fs)
    return sorted(fs)
