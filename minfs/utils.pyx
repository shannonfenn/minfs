# distutils: language = c++
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: initializedcheck=False
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


def valid_feature_set(packed_type_t[:, :] coverage, size_t Np, set F):
    cdef:
        size_t f
        np.ndarray[packed_type_t, ndim=1] covered_pairs

    # to mask pairs off later in F->P lookup
    covered_pairs = np.zeros_like(coverage[0])
    for f in F:
        np.bitwise_or(covered_pairs, coverage[f], covered_pairs)
    
    return bc.popcount_vector(covered_pairs) == Np


cpdef packed_coverage_map(np.uint8_t[:, :] X, np.uint8_t[:] y):
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
    return coverage, Np


cpdef dual_packed_coverage_maps(np.uint8_t[:, :] X, np.uint8_t[:] y):
    cdef:
        size_t Np, Nf, i, i1, f, p, num_p_chunks, num_f_chunks, fchunk, pchunk
        packed_type_t pmask, fmask
        packed_type_t[:, :] PF, FP
        vector[size_t] class_0_indices, class_1_indices
        np.uint8_t[:] pattern0, pattern1
        np.uint8_t unequal

    Nf = X.shape[1]
    for i in range(y.shape[0]):
        if y[i]:
            class_1_indices.push_back(i)
        else:
            class_0_indices.push_back(i)

    Np = class_0_indices.size() * class_1_indices.size()
    num_p_chunks = int(ceil(Np / <double>PACKED_SIZE))
    num_f_chunks = int(ceil(Nf / <double>PACKED_SIZE))
    # build packed coverages
    PF = np.zeros((Np, num_f_chunks), dtype=packed_type)
    FP = np.zeros((Nf, num_p_chunks), dtype=packed_type)

    p = 0
    pchunk = 0
    pmask = 1
    for i in class_0_indices:
        pattern0 = X[i]
        for i1 in class_1_indices:
            pattern1 = X[i1]
            fchunk = 0
            fmask = 1
            for f in range(Nf):
                if pattern0[f] != pattern1[f]:
                    PF[p, fchunk] |= fmask
                    FP[f, pchunk] |= pmask
                fmask <<= 1
                fchunk += (fmask == 0)
                fmask += (fmask == 0)
            pmask <<= 1
            pchunk += (pmask == 0)
            pmask += (pmask == 0)

            p += 1
    return PF, FP


cpdef redundant_features(packed_type_t[:, :] C, set F):
    cdef:
        size_t f
        np.ndarray[packed_type_t, ndim=1] covered, doubly_covered, temp
        list redundant

    # to mask pairs off later in F->P lookup
    covered = np.zeros_like(C[0])
    doubly_covered = np.zeros_like(C[0])
    temp = np.empty_like(C[0])
    for f in F:
        # mark any already covered features as doubly covered
        np.bitwise_and(covered, C[f], temp)
        np.bitwise_or(temp, doubly_covered, doubly_covered)
        # mark newly covered features as covered
        np.bitwise_or(covered, C[f], covered)

    # redundant features will only cover pairs that are doubly covered
    redundant = []
    for f in F:
        np.bitwise_and(doubly_covered, C[f], covered)
        if np.array_equal(C[f], covered):
            redundant.append(f)

    return redundant


cpdef remove_redundant_features(packed_type_t[:, :] C, size_t Np, set F):
    cdef:
        size_t f
        list redundant

    redundant = redundant_features(C, F)

    while(redundant):
        f = random.choice(redundant)
        F.discard(f)
        redundant = redundant_features(C, F)


cpdef best_repair_features(packed_type_t[:, :] C, set F):
    cdef:
        size_t f, Nf, best_num_covered, num_covered, num_uncovered
        np.ndarray[packed_type_t, ndim=1] covered_pairs, uncovered_pairs
        set best

    Nf = C.shape[0]
        
    # to mask pairs off later in F->P lookup
    covered_pairs = np.zeros_like(C[0])
    uncovered_pairs = np.empty_like(covered_pairs)
    for f in F:
        np.bitwise_or(covered_pairs, C[f], covered_pairs)
    np.invert(covered_pairs, uncovered_pairs)

    num_uncovered = bc.popcount_vector(uncovered_pairs)
    
    # find feature which covers the most uncovered features
    best_num_covered = 0
    best = set()
    for f in range(Nf):
        # no need to check if f in F since by definition none of the pairs are covered by F
        np.bitwise_and(uncovered_pairs, C[f], covered_pairs)
        num_covered = bc.popcount_vector(covered_pairs)
        if num_covered > best_num_covered:
            best_num_covered = num_covered
            best = {f}
        elif num_covered == best_num_covered and num_covered > 0:
            best.add(f)

    return best, best_num_covered, num_uncovered - best_num_covered


def greedy_repair(packed_type_t[:, :] C, size_t Np, set fs, bint verbose=False):
    while(not valid_feature_set(C, Np, fs)):
        best_f, Nc, Nuc = best_repair_features(C, fs)
        f = random.choice(tuple(best_f))
        fs.add(f)
        if verbose:
            print(Nc, Nuc, len(fs), '\trandomly chose {} from {}'.format(f, len(best_f)))
    return fs
