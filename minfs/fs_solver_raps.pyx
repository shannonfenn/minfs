# distutils: language = c++
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: initializedcheck=False
import numpy as np
cimport numpy as np
cimport bitpacking.bitcount as bc
import minfs.utils as utils
import random

from bitpacking.packing cimport packed_type_t
from bitpacking.packing cimport PACKED_SIZE
from bitpacking.packing import packed_type


def construction(packed_type_t[:, :] C, size_t Np, priority, restriction):
    assert 0 <= priority <= 1 and 0 <= restriction <= 1
    fs = set()
    while(not utils.valid_feature_set(C, Np, fs)):
        CL, Ncov, _ = utils.best_repair_features(C, fs)
        if random.random() >= priority:
            CL = candidate_features(C, Ncov * (1 - restriction))
        f = random.choice(tuple(CL))
        fs.add(f)
        # if verbose:
        #     print(Nc, Nuc, len(fs), '\trandomly chose {} from {}'.format(f, len(best_f)))
    utils.remove_redundant_features(C, Np, fs)
    return fs


cpdef candidate_features(packed_type_t[:, :] C, double threshold):
    cdef:
        size_t f, Nf, num_covered
        list candidates = []

    Nf = C.shape[0]

    for f in range(Nf):
        num_covered = bc.popcount_vector(C[f])
        if num_covered >= threshold:
            candidates.append(f)

    return candidates


def local_search(packed_type_t[:, :] C, size_t Np, set F, imp_iterations, search_magnitude):
    assert 0 < imp_iterations and 0 <= search_magnitude <= 1

    for i in range(imp_iterations):
        num_remove = min(1, int(search_magnitude * len(F)))

        # remove this many features
        F_new

        # construct induced subproblem
        C_sub, Np_sub

        # solve induced subproblem and accumulate
        F_sub = construction(C_sub, Np_sub, priority, restriction)

        # include kept features
        F_new |= F_sub

        utils.remove_redundant_features(C, Np, F_new)

        # replace if as good
        if len(F_new) <= len(F):
            F = F_new
    return F


def minfs(X, y, priority, restriction):
    X = X.astype(np.uint8)
    y = y.astype(np.uint8)

    if np.all(y) or not np.any(y):
        # constant target - no solutions
        return []

    C, Np = utils.packed_coverage_map(X, y)
    fs = construction(C, Np, priority, restriction)
    return list(fs)
