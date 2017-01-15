# distutils: language = c++
# cython: boundscheck=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: wraparound=False
# cython: initializedcheck=False

# cython: profile=True

import numpy as np
cimport numpy as np
import minfs.utils as utils
cimport minfs.utils as utils
import random

cimport bitpacking.bitcount as bc
cimport bitpacking.packing as pk

from bitpacking.packing cimport packed_type_t
from bitpacking.packing import packed_type


cdef class SetCoverSolution:
    cdef:
        public set features
        public utils.SetCoverInstance instance
        public size_t[:] num_will_cover
        public packed_type_t[:] covered, doubly_covered, uncovered, temp, temp2, end_mask

    def __init__(self, instance):
        self.features = set()
        self.instance = instance
        self.num_will_cover = np.zeros(instance.C.shape[0], dtype=np.uintp)
        self.covered = np.zeros(instance.C.shape[1], dtype=packed_type)
        self.doubly_covered = np.zeros_like(self.covered)
        self.uncovered = np.bitwise_not(self.covered)
        self.temp = np.zeros_like(self.covered)
        self.temp2 = np.zeros_like(self.covered)
        self.end_mask = np.ones_like(self.covered)
        self.end_mask[self.end_mask.shape[0] - 1] = pk.generate_end_mask(instance.Np)

    cpdef set_features(self, set new_features):
        cdef:
            size_t f
            np.ndarray[packed_type_t, ndim=2] C_np
            np.ndarray[packed_type_t] covered_np = np.asarray(self.covered),
            np.ndarray[packed_type_t] uncovered_np = np.asarray(self.uncovered)

        # wrap the memviews with numpy arrays
        C_np = np.asarray(self.instance.C)
        covered_np = np.asarray(self.covered)
        # add the features
        self.features = new_features
        # re-calculate covered pairs
        self.covered[...] = 0
        for f in self.features:
            np.bitwise_or(covered_np, C_np[f], covered_np)
        np.bitwise_not(covered_np, uncovered_np)

    cpdef add_features(self, set new_features):
        cdef:
            size_t f
            np.ndarray[packed_type_t, ndim=2] C_np = np.asarray(self.instance.C)
            np.ndarray[packed_type_t] covered_np = np.asarray(self.covered),
            np.ndarray[packed_type_t] uncovered_np = np.asarray(self.uncovered)
        # add the features
        self.features |= new_features
        # mark of any newly covered pairs
        for f in new_features:
            np.bitwise_or(covered_np, C_np[f], covered_np)
        np.bitwise_not(covered_np, uncovered_np)

    cpdef remove_features(self, set features_to_remove):
        cdef:
            size_t f
            np.ndarray[packed_type_t, ndim=2] C_np = np.asarray(self.instance.C)
            np.ndarray[packed_type_t] covered_np = np.asarray(self.covered),
            np.ndarray[packed_type_t] uncovered_np = np.asarray(self.uncovered)
        # remove the features
        self.features -= features_to_remove
        # re-calculate covered pairs
        self.covered[...] = 0
        for f in self.features:
            np.bitwise_or(covered_np, C_np[f], covered_np)
        np.bitwise_not(covered_np, uncovered_np)

    cpdef _remove_feature(self, size_t f):
        # THIS METHOD RELIES ON DOUBLY_COVERED BEING SET CORRECTLY
        cdef:
            np.ndarray[packed_type_t, ndim=2] C_np = np.asarray(self.instance.C)
            np.ndarray[packed_type_t] covered_np = np.asarray(self.covered),
            np.ndarray[packed_type_t] uncovered_np = np.asarray(self.uncovered)
            np.ndarray[packed_type_t] doubly_covered_np = np.asarray(self.doubly_covered),
            np.ndarray[packed_type_t] temp_np = np.asarray(self.temp),
            np.ndarray[packed_type_t] temp2_np = np.asarray(self.temp2)
        # temp  : the inverse of C[f]
        # temp2 : mask for updating covered

        # remove the feature - will raise an error if feature not present (desired - since this
        # would break the coverage update below)
        self.features.remove(f)
        # get mask for which features are not left uncovered
        np.bitwise_not(C_np[f], temp_np)
        np.bitwise_or(temp_np, doubly_covered_np, temp2_np)
        # apply mask 
        np.bitwise_and(temp2_np, covered_np, covered_np)
        np.bitwise_not(covered_np, uncovered_np)


    cpdef remove_redundant(self):
        redundant = utils.get_redundant_feature(
            self.instance.C, self.features, self.doubly_covered, self.temp, self.temp2)

        while redundant is not None:
            self._remove_feature(redundant)
            redundant = utils.get_redundant_feature(
                self.instance.C, self.features, self.doubly_covered, self.temp, self.temp2)

    cpdef satisfied(self):
        return bc.popcount_vector(self.covered) == self.instance.Np


cpdef repair(SetCoverSolution soln, double priority, double restriction):
    cdef:
        np.ndarray[packed_type_t, ndim=2] C_np
        np.ndarray[packed_type_t] uncovered_np, temp_np
        np.ndarray[size_t] num_will_cover_np
        size_t f, Nf, max_covered = 0
        list candidates

    C_np = np.asarray(soln.instance.C)
    uncovered_np = np.asarray(soln.uncovered)
    temp_np = np.asarray(soln.temp)
    num_will_cover_np = np.asarray(soln.num_will_cover)

    assert 0 <= priority <= 1 and 0 <= restriction <= 1
    Nf = soln.instance.Nf
    while(not soln.satisfied()):
        # count number of uncovered pairs that each feature would cover
        for f in range(Nf):
            # no need to check if f in current FS since by definition none of the pairs are covered
            np.bitwise_and(uncovered_np, C_np[f], temp_np)
            num_will_cover_np[f] = bc.popcount_vector(soln.temp)
        max_covered = num_will_cover_np.max()
    
        threshold = max_covered
        if random.random() >= priority:
            threshold *= (1 - restriction)
        # build candidate list
        candidates = [f for f in range(Nf) if num_will_cover_np[f] >= threshold]
        f = random.choice(candidates)
        soln.add_features({f})
    soln.remove_redundant()


# # def induced_subproblem(SetCoverSolution soln):
# #     cdef:
# #         size_t f
# #         np.ndarray[packed_type_t, ndim=1] remaining_pairs
# #         np.ndarray[packed_type_t, ndim=2] C
# #         list remaining_features

# #     C = np.asarray(soln.instance.C)

# #     remaining_features = list(set(range(C.shape[0])) - F)
# #     C_sub = C[remaining_features]

# #     soln.temp[...] = 0
# #     remaining_pairs = np.asarray(soln.temp)

# #     for f in remaining_features:
# #         np.bitwise_or(remaining_pairs, C[f], remaining_pairs)
# #     Np_sub = bc.popcount_vector(soln.temp) # use the memview interface

# #     inst_sub = utils.SetCoverInstance(C_sub, Np_sub)
# #     return inst_sub, remaining_features


def local_search(soln, imp_iterations, search_magnitude, priority, restriction):
    assert 0 < imp_iterations and 0 <= search_magnitude <= 1

    for i in range(imp_iterations):
        # print('LS iteration', i)
        # ensure at least 1 feature is replaced
        num_remove = max(1, int(search_magnitude * len(soln.features)))

        F_before = set(soln.features)

        # keep only this many features
        features_to_remove = set(random.sample(soln.features, num_remove))
        soln.remove_features(features_to_remove)

        # solve induced subproblem (equivalent to just repairing)
        repair(soln, priority, restriction)
        soln.remove_redundant()

        # roll back if worse
        if len(soln.features) > len(F_before):
            soln.set_features(F_before)


def _solve(instance, iterations, improvement_iterations, search_magnitude,
           priority, restriction, improvement):
    min_k_constructed = instance.Nf
    best = set(range(min_k_constructed))
    soln = SetCoverSolution(instance)
    for i in range(iterations):
        # print('main iteration', i)

        # Start from scratch
        soln.set_features(set())
        # Build initial feasible solution
        repair(soln, priority, restriction)
        min_k_constructed = min(min_k_constructed, len(soln.features))
        # if this is nearly as small as the smallest seen after construction
        if len(soln.features) <= (1 + improvement) * min_k_constructed:
            local_search(soln, improvement_iterations, search_magnitude, priority, restriction) 
        if len(soln.features) < len(best):
            best = set(soln.features)
    return best


def single_minimum_feature_set(X, y, iterations=20, improvement_iterations=100, search_magnitude=0.3,
          priority=0.05, restriction=0.15, improvement=0.15):
    X = X.astype(np.uint8)
    y = y.astype(np.uint8)

    if np.all(y) or not np.any(y):
        # constant target - no solutions
        return []

    instance = utils.build_instance(X, y)

    F = _solve(instance, iterations, improvement_iterations, search_magnitude,
               priority, restriction, improvement)

    return list(F)
