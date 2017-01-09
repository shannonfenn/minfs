# distutils: language = c++
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: initializedcheck=False
# cython: profile=True


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


class PackedCoverageMatrix(np.ndarray):
    def __new__(cls, input_array, Np):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.Np = Np
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        try:
            self.Np = obj.Np
        except:
            self.Np = None

    def __reduce__(self):
        # Get the parent's __reduce__ tuple
        pickled_state = super(PackedCoverageMatrix, self).__reduce__()
        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (self.Np, )
        # Return a tuple that replaces the parent's __setstate__ with our own
        return (pickled_state[0], pickled_state[1], new_state)

    @cython.wraparound(True) # turn on wraparound for this class
    def __setstate__(self, state):
        self.Np = state[-1]  # Set the info attribute
        # Call the parent's __setstate__ with the other tuple elements.
        super(PackedCoverageMatrix, self).__setstate__(state[0:-1])


def valid_feature_set(np.ndarray[packed_type_t, ndim=2] coverage, set F):
    cdef:
        size_t f
        np.ndarray[packed_type_t, ndim=1] covered_pairs

    # to mask pairs off later in F->P lookup
    covered_pairs = np.zeros_like(coverage[0])
    for f in F:
        np.bitwise_or(covered_pairs, coverage[f], covered_pairs)
    
    return bc.popcount_vector(covered_pairs) == coverage.Np


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
    return PackedCoverageMatrix(coverage, Np)


cdef bint _arr_eq(np.ndarray[packed_type_t, ndim=1] x, np.ndarray[packed_type_t, ndim=1] y):
    cdef:
        size_t i

    for i in range(x.shape[0]):
        if x[i] != y[i]:
            return False
    return True


cpdef redundant_features(np.ndarray[packed_type_t, ndim=2] C, set F):
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
        # if np.array_equal(C[f], covered):
        if _arr_eq(C[f], covered):
            redundant.append(f)

    return redundant


cpdef remove_redundant_features(np.ndarray[packed_type_t, ndim=2] C, set F):
    cdef:
        size_t f
        list redundant

    redundant = redundant_features(C, F)

    while(redundant):
        f = random.choice(redundant)
        F.discard(f)
        redundant = redundant_features(C, F)


cpdef best_repair_features(np.ndarray[packed_type_t, ndim=2] C, set F):
    cdef:
        size_t f, Nf, best_num_covered, num_covered, num_uncovered
        np.ndarray[packed_type_t, ndim=1] covered_pairs, uncovered_pairs
        packed_type_t[:] covered_pairs_memvw, uncovered_pairs_memvw
        list best

    Nf = C.shape[0]
        
    uncovered_pairs = np.empty_like(C[0])
    covered_pairs = np.zeros_like(C[0])
    uncovered_pairs_memvw = uncovered_pairs
    covered_pairs_memvw = covered_pairs
    
    for f in F:
        np.bitwise_or(covered_pairs, C[f], covered_pairs)
    np.invert(covered_pairs, uncovered_pairs)

    num_uncovered = bc.popcount_vector(uncovered_pairs_memvw) - (PACKED_SIZE - C.Np % PACKED_SIZE) % PACKED_SIZE

    # find feature which covers the most uncovered features
    best_num_covered = 0
    best = []
    for f in range(Nf):
        # no need to check if f in F since by definition none of the pairs are covered by F
        np.bitwise_and(uncovered_pairs, C[f], covered_pairs)
        num_covered = bc.popcount_vector(covered_pairs_memvw)
        if num_covered > best_num_covered:
            best_num_covered = num_covered
            best = [f]
        elif num_covered == best_num_covered and num_covered > 0:
            best.append(f)

    return best, best_num_covered, num_uncovered - best_num_covered


def greedy_repair(np.ndarray[packed_type_t, ndim=2] C, set fs, bint verbose=False):
    while(not valid_feature_set(C, fs)):
        best_f, Nc, Nuc = best_repair_features(C, fs)
        f = random.choice(tuple(best_f))
        fs.add(f)
        if verbose:
            print(Nc, Nuc, len(fs), '\trandomly chose {} from {}'.format(f, len(best_f)))
    return fs


def induced_subproblem(np.ndarray[packed_type_t, ndim=2] C, set F):
    cdef:
        size_t f
        np.ndarray[packed_type_t, ndim=1] remaining_pairs
        list remaining_features

    remaining_features = list(set(range(C.shape[0])) - F)
    C_sub = C[remaining_features]
    remaining_pairs = np.zeros_like(C[0])
    for f in remaining_features:
        np.bitwise_or(remaining_pairs, C[f], remaining_pairs)
    Np_sub = bc.popcount_vector(remaining_pairs)
    return PackedCoverageMatrix(C_sub, Np_sub), remaining_features
