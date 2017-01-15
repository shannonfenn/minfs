import numpy as np
cimport numpy as np

from bitpacking.packing cimport packed_type_t

cdef class SetCoverInstance:
    cdef:
        public size_t Np, Nf
        public packed_type_t[:,:] C

# def is_valid_fs(np.ndarray[packed_type_t, ndim=2] coverage, set F):

cpdef build_instance(np.uint8_t[:, :] X, np.uint8_t[:] y)

cdef bint _arr_eq(np.ndarray[packed_type_t] x, np.ndarray[packed_type_t] y)

cpdef packed_type_t[:] get_doubly_covered(
        packed_type_t[:, :] C, set F, packed_type_t[:] doubly_covered=*,
        packed_type_t[:] covered_tracker=*, packed_type_t[:] update_mask=*)

cpdef get_redundant_feature(packed_type_t[:, :] C, set F, packed_type_t[:] doubly_covered=*,
                            packed_type_t[:] covered_tracker=*, packed_type_t[:] update_mask=*)


# cpdef best_repair_features(np.ndarray[packed_type_t, ndim=2] C, set F,
#                            packed_type_t[:] covered=*, packed_type_t[:] uncovered=*)


# def greedy_repair(np.ndarray[packed_type_t, ndim=2] C, set fs, bint verbose=False)