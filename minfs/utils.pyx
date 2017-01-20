# distutils: language = c++
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: initializedcheck=False

# cython: profile=False

import cython
import numpy as np
cimport numpy as np
cimport bitpacking.packing as pk
cimport bitpacking.bitcount as bc
from libc.math cimport ceil
from libcpp.vector cimport vector
import random

from bitpacking.packing cimport packed_type_t
from bitpacking.packing cimport PACKED_SIZE
from bitpacking.packing import packed_type


cdef class SetCoverInstance:
    def __init__(self, C, Np):
        self.C = np.array(C, dtype=packed_type)
        self.Np = Np
        self.Nf = self.C.shape[0]





# def is_valid_fs(np.ndarray[packed_type_t, ndim=2] coverage, set F):
#     cdef:
#         size_t f
#         np.ndarray[packed_type_t, ndim=1] covered_pairs

#     # to mask pairs off later in F->P lookup
#     covered_pairs = np.zeros_like(coverage[0])
#     for f in F:
#         np.bitwise_or(covered_pairs, coverage[f], covered_pairs)
    
#     return bc.popcount_vector(covered_pairs) == coverage.Np


cpdef build_instance(np.uint8_t[:, :] X, np.uint8_t[:] y):
    cdef:
        size_t Np, Nf, i, i1, f, num_chunks, chunk
        packed_type_t mask
        packed_type_t[:, :] coverage
        vector[size_t] class_0_indices, class_1_indices
        np.uint8_t[:] pattern0, pattern1

    Nf = X.shape[1]
    for i in range(y.shape[0]):
        if y[i]:
            class_1_indices.push_back(i)
        else:
            class_0_indices.push_back(i)

    Np = class_0_indices.size() * class_1_indices.size()
    num_chunks = int(ceil(Np / <double>PACKED_SIZE))
    # build packed coverages
    coverage = np.zeros((Nf, num_chunks), dtype=packed_type)

    chunk = 0
    mask = 1
    for i in class_0_indices:
        pattern0 = X[i]
        for i1 in class_1_indices:
            pattern1 = X[i1]
            for f in range(Nf):
                if pattern0[f] != pattern1[f]:
                    coverage[f, chunk] |= mask
            mask <<= 1
            # handle rollover into new chunk
            chunk += (mask == 0)
            mask += (mask == 0)
    return SetCoverInstance(coverage, Np)


cdef bint _arr_eq(np.ndarray[packed_type_t] x, np.ndarray[packed_type_t] y):
    cdef:
        size_t i

    for i in range(x.shape[0]):
        if x[i] != y[i]:
            return False
    return True


cpdef packed_type_t[:] get_doubly_covered(
        packed_type_t[:, :] C, set F, packed_type_t[:] doubly_covered=None,
        packed_type_t[:] covered_tracker=None, packed_type_t[:] update_mask=None):
    # redundant features will only cover pairs that are doubly covered
    cdef:
        size_t f
        np.ndarray[packed_type_t, ndim=2] C_np
        np.ndarray[packed_type_t, ndim=1] doubly_covered_np, covered_tracker_np, update_mask_np

    C_np = np.asarray(C)

    # setup - handle optionally provided memviews
    if doubly_covered is None:
        doubly_covered_np = np.zeros_like(C_np[0])
    else:
        doubly_covered[...] = 0
        doubly_covered_np = np.asarray(doubly_covered)
    if covered_tracker is None:
        covered_tracker_np = np.zeros_like(C_np[0])
    else:
        covered_tracker[...] = 0
        covered_tracker_np = np.asarray(covered_tracker)
    if update_mask is None:
        update_mask_np = np.empty_like(C_np[0])
    else:
        update_mask_np = np.asarray(update_mask)

    # calculate which pairs are covered by 2 or more features
    for f in F:
        # covered_tracker_np - tracker for covered features
        # update_mask_np - holder for pairs the current feature covers which are already covered
        # mask of pairs covered by this feature that are already covered
        np.bitwise_and(covered_tracker_np, C_np[f], update_mask_np)
        # mark these pairs as doubly covered
        np.bitwise_or(update_mask_np, doubly_covered_np, doubly_covered_np)
        # mark newly covered pairs as covered
        np.bitwise_or(covered_tracker_np, C_np[f], covered_tracker_np)

    return doubly_covered


cpdef get_redundant_feature(packed_type_t[:, :] C, set F, packed_type_t[:] doubly_covered=None,
                            packed_type_t[:] covered_tracker=None, packed_type_t[:] update_mask=None):
    cdef:
        size_t f
        np.ndarray[packed_type_t, ndim=2] C_np
        np.ndarray[packed_type_t, ndim=1] doubly_covered_np, covered_tracker_np
    
    doubly_covered = get_doubly_covered(C, F, doubly_covered, covered_tracker, update_mask)

    C_np = np.asarray(C)
    doubly_covered_np = np.asarray(doubly_covered)  # will have been created by call above
    if covered_tracker is None:
        covered_tracker_np = np.zeros_like(C_np[0])
    else:
        covered_tracker[...] = 0
        covered_tracker_np = np.asarray(covered_tracker)

    # We need a second pass
    for f in F:
        np.bitwise_and(doubly_covered_np, C_np[f], covered_tracker_np)
        if _arr_eq(C_np[f], covered_tracker_np):
            return f
    return None


# cpdef best_repair_features(np.ndarray[packed_type_t, ndim=2] C, set F,
#                            packed_type_t[:] covered=None, packed_type_t[:] uncovered=None):
#     cdef:
#         size_t f, Nf, best_num_covered, num_covered, num_uncovered
#         np.ndarray[packed_type_t, ndim=1] covered_np, uncovered_np
#         list best

#     Nf = C.shape[0]
        
#     if covered is None:
#         covered = np.zeros_like(C[0])
#     if uncovered is None:
#         uncovered = np.empty_like(C[0])
    
#     # wrap np arrays around memviews
#     covered_np = np.asarray(covered)
#     uncovered_np = np.asarray(uncovered)

#     for f in F:
#         np.bitwise_or(covered_np, C[f], covered_np)
#     np.invert(covered_np, uncovered_np)

#     num_uncovered = bc.popcount_vector(uncovered) - (PACKED_SIZE - C.Np % PACKED_SIZE) % PACKED_SIZE

#     # find feature which covers the most uncovered features
#     best_num_covered = 0
#     best = []
#     for f in range(Nf):
#         # no need to check if f in F since by definition none of the pairs are covered by F
#         np.bitwise_and(uncovered_np, C[f], covered_np)
#         num_covered = bc.popcount_vector(covered)
#         if num_covered > best_num_covered:
#             best_num_covered = num_covered
#             best = [f]
#         elif num_covered == best_num_covered and num_covered > 0:
#             best.append(f)

#     return best, best_num_covered, num_uncovered - best_num_covered


# def greedy_repair(np.ndarray[packed_type_t, ndim=2] C, set fs, bint verbose=False):
#     while(not valid_feature_set(C, fs)):
#         best_f, Nc, Nuc = best_repair_features(C, fs)
#         f = random.choice(tuple(best_f))
#         fs.add(f)
#         if verbose:
#             print(Nc, Nuc, len(fs), '\trandomly chose {} from {}'.format(f, len(best_f)))
#     return fs
