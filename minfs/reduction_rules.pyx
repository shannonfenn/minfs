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


cdef set rule1(packed_type_t[:, :] PFcov, packed_type_t[:, :] FPcov, set F, set P):
    cdef:
        size_t j, p, f
        packed_type_t[:] Fmask, Pmask, row
        set forced

    forced = set()

    Fmask = np.zeros_like(PFcov[0])

    for p in P:
        row = PFcov[p]
        if bc.popcount_vector(row) == 1:
            forced.add(bc.scan_vector(row))
            for j in range(Fmask.shape[0]):
                Fmask[j] |= row[j]

    # to mask pairs off later in F->P lookup
    Pmask = np.zeros_like(FPcov[0])
    for f in forced:
        for j in range(Pmask.shape[0]):
            Pmask[j] |= FPcov[f, j]

    P.difference_update(np.flatnonzero(pk.unpackvec(Pmask, Pmask.shape[0]*pk.PACKED_SIZE)))

    F -= forced

    for f in F:
        for j in range(Pmask.shape[0]):
            FPcov[f, j] &= ~Pmask[j]

    # since every pair covered by the discovered features was removed,
    # we don't need to mask any of the discovered features out of P->F

    return forced


cdef bint rule2(packed_type_t[:, :] PFcov, packed_type_t[:, :] FPcov, set F, set P):
    cdef:
        size_t i, i2, j, f1, f2, p, f, Nf, Nch
        bint sub
        packed_type_t[:] Fmask
        size_t[:] Flist, counts, order
        set redundant_features

    Nf = len(F)
    Nch = FPcov.shape[1]

    Flist = np.empty(Nf, dtype=np.uintp)
    counts = np.empty(Nf, dtype=np.uintp)
    i = 0
    for f in F:
        counts[i] = bc.popcount_vector(FPcov[f, :])
        Flist[i] = f
        i += 1
    order = np.argsort(counts).astype(np.uintp)

    redundant_features = set()

    i = 0
    while i < Nf:
        f1 = Flist[order[i]]
        i2 = i+1
        while i2 < Nf:
            f2 = Flist[order[i2]]
            # if examples covered by f1 are a subset of those covered by f2
            # then f1 is a redundant feature since f2 covers all its pairs
            sub = True
            for j in range(Nch):
                if FPcov[f1, j] & ~FPcov[f2, j] != 0:
                    sub = False
                    break
            if sub:
                redundant_features.add(f1)
                break
            i2 += 1
        i += 1

    # ignore features in F->P map
    F -= redundant_features

    # mask off features in P->F map
    Fmask = np.zeros_like(PFcov[0])
    pk.setbits(Fmask, F)
    for p in P:
        for j in range(Fmask.shape[0]):
            PFcov[p, j] &= Fmask[j]

    return len(redundant_features) > 0


cdef bint rule3(packed_type_t[:, :] PFcov, packed_type_t[:, :] FPcov, set F, set P):
    cdef:
        size_t i, i2, j, p1, p2, p, f, Np, Nch
        bint sub
        packed_type_t[:] Pmask
        size_t[:] Plist, counts, order
        set redundant_pairs

    Np = len(P)
    Nch = PFcov.shape[1]

    Plist = np.empty(Np, dtype=np.uintp)
    counts = np.empty(Np, dtype=np.uintp)
    i = 0
    for p in P:
        counts[i] = bc.popcount_vector(PFcov[p, :])
        Plist[i] = p
        i += 1
    order = np.argsort(counts).astype(np.uintp)[::-1]
    
    redundant_pairs = set()

    i = 0
    while i < Np:
        p1 = Plist[order[i]]
        i2 = i+1
        while i2 < Np:
            p2 = Plist[order[i2]]
            # if features covering p2 are a subset of those covering p1
            # then p1 is redundant since any feature that covers p2 will cover p1
            sub = True
            for j in range(Nch):
                if ~PFcov[p1, j] & PFcov[p2, j] != 0:
                    sub = False
                    break
            if sub:
                redundant_pairs.add(p1)
                break
            i2 += 1
        i += 1

    # ignore discovered pairs in P->F map
    P -= redundant_pairs

    # mask off pairs in F->P map
    Pmask = np.zeros_like(FPcov[0])
    pk.setbits(Pmask, P)
    for f in F:
        for j in range(Pmask.shape[0]):
            FPcov[f, j] &= Pmask[j]

    return len(redundant_pairs) > 0


cpdef apply_reduction_rules(packed_type_t[:, :] PFcov, packed_type_t[:, :] FPcov,
                            bint apply1=True, bint apply2=True, bint apply3=True):
    cdef:
        bint finished, r1_applied, r2_applied, r3_applied
        size_t Np, Nf
        set P, F, Fin, forced

    finished = False
    r1_applied = apply1
    r2_applied = apply2
    r3_applied = apply3

    Np = PFcov.shape[0]
    Nf = FPcov.shape[0]
    P = set(range(Np))
    F = set(range(Nf))
    Fin = set()

    while(not finished):
        if apply1:
            forced = rule1(PFcov, FPcov, F, P)
            r1_applied = len(forced) > 0
            Fin |= forced
            # print('RR1\n  applied: {}\n  forced: {}\n  F: {}\n  P: {}'.format(r1_applied, forced, F, P))
        finished = not (r1_applied or r2_applied or r3_applied)
        if apply2 and not finished:
            r2_applied = rule2(PFcov, FPcov, F, P)
            # print('RR2\n  applied: {}\n  F: {}\n  P: {}'.format(r2_applied, F, P))
        finished = not (r1_applied or r2_applied or r3_applied)
        if apply3 and not finished:
            r3_applied = rule3(PFcov, FPcov, F, P)
            # print('RR3\n  applied: {}\n  F: {}\n  P: {}'.format(r3_applied, F, P))
        finished = not (r1_applied or r2_applied or r3_applied)
        
    return Fin, F, P
