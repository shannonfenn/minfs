import numpy as np
import bitpacking.bitcount as bc
import minfs.utils as utils
import random


def construction(C, priority, restriction):
    assert 0 <= priority <= 1 and 0 <= restriction <= 1
    fs = set()
    Nf = C.shape[0]
    while(not utils.valid_feature_set(C, fs)):
        candidates, Ncov, _ = utils.best_repair_features(C, fs)
        if random.random() >= priority:
            threshold = Ncov * (1 - restriction)
            candidates = [f for f in range(Nf) if bc.popcount_vector(C[f]) >= threshold]
        f = random.choice(candidates)
        fs.add(f)
    utils.remove_redundant_features(C, fs)
    return fs


def local_search(C, F, imp_iterations, search_magnitude, priority, restriction):
    assert 0 < imp_iterations and 0 <= search_magnitude <= 1

    for i in range(imp_iterations):
        # print('LS iteration', i)
        # ensure at least 1 feature is replaced
        num_remove = max(1, int(search_magnitude * len(F)))
        num_keep = len(F) - num_remove

        # keep only this many features
        F_new = set(random.sample(F, num_keep))

        # construct induced subproblem
        C_sub, index = utils.induced_subproblem(C, F_new)

        # solve induced subproblem and accumulate
        F_sub = construction(C_sub, priority, restriction)
        # get real feature indices
        F_sub = set(index[f] for f in F_sub)

        # include kept features
        F_new |= F_sub

        utils.remove_redundant_features(C, F_new)

        # replace if as good
        if len(F_new) <= len(F):
            F = F_new
    return F


def _solve(C, iterations, improvement_iterations, search_magnitude,
           priority, restriction, improvement):
    min_k_constructed = C.shape[0]
    best = set(range(min_k_constructed))
    for i in range(iterations):
        # print('main iteration', i)
        F = construction(C, priority, restriction)
        k = len(F)
        min_k_constructed = min(min_k_constructed, k)
        # if this is nearly as small as the smallest seen after construction
        if k <= (1 + improvement) * min_k_constructed:
            F = local_search(C, F, improvement_iterations, search_magnitude, priority, restriction)
        if k < len(best):
            best = F
    return best


def single_minimum_feature_set(X, y, iterations=20, improvement_iterations=100, search_magnitude=0.3,
          priority=0.05, restriction=0.15, improvement=0.15):
    X = X.astype(np.uint8)
    y = y.astype(np.uint8)

    if np.all(y) or not np.any(y):
        # constant target - no solutions
        return []

    C = utils.packed_coverage_map(X, y)

    F = _solve(C, iterations, improvement_iterations, search_magnitude,
               priority, restriction, improvement)

    return list(F)
