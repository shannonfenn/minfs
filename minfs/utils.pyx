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

