# from boolnet.learning.feature_selection import abk_file, minimum_feature_set
import minfs.reduction_rules as rr
import minfs.utils as utils
import bitpacking.packing as pk
import numpy as np
from numpy.testing import assert_array_equal
import pytest


@pytest.fixture(params=[1, 2, 3, 4, 5])
def instance(request):
    filename_base = 'minfs/test/instances/{}'.format(request.param)
    with np.load(filename_base + '.npz') as data:
        features = data['features']
        target = data['target']
        feature_sets = data['feature_sets']
    return features, target, feature_sets


@pytest.mark.parametrize('apply1', [False, True])
@pytest.mark.parametrize('apply2', [False, True])
@pytest.mark.parametrize('apply3', [False, True])
def test_packed_method(instance, apply1, apply2, apply3):
    X, y, _ = instance
    C = coverage(X, y)

    Np, Nf = C.shape

    forced_, F_, P_ = apply(C, apply1, apply2, apply3)

    forced_ = set(forced_)
    Csub_ = C[sorted(list(P_)), :][:, sorted(list(F_))]

    PFcov, FPcov = utils.dual_packed_coverage_maps(X, y)

    forced, F, P = rr.apply_reduction_rules(PFcov, FPcov, apply1, apply2, apply3)

    Csub1 = pk.unpackmat(PFcov, Nf, transpose=False)[sorted(list(P)), :][:, sorted(list(F))]
    Csub2 = pk.unpackmat(FPcov, Np, transpose=True)[sorted(list(P)), :][:, sorted(list(F))]

    assert forced == forced_
    assert F == F_
    assert P == P_
    assert_array_equal(Csub_, Csub1)
    assert_array_equal(Csub_, Csub2)


def coverage(features, target):
    Ne, Nf = features.shape
    class_0_indices = np.flatnonzero(target == 0)
    class_1_indices = np.flatnonzero(target)
    num_eg_pairs = class_0_indices.size * class_1_indices.size
    coverage_matrix = np.zeros((num_eg_pairs, Nf), dtype=np.uint8)
    i = 0
    for i0 in class_0_indices:
        for i1 in class_1_indices:
            np.not_equal(features[i0], features[i1], coverage_matrix[i])
            i += 1
    return coverage_matrix


# Ground truth methods

def reduction1(C):
    # returns: F - features that must be in the solution
    #          P - pairs which can be removed from the sub-problem
    # if this pair is covered by only one feature - mark it for removal
    candidate_pairs = (C.sum(axis=1) == 1)
    # the features to keep will be any that differ for the candidate pairs
    F = np.flatnonzero(C[candidate_pairs].sum(axis=0))
    # get all pairs covered by these features
    P = np.flatnonzero(C[:, F].sum(axis=1))
    return F, P


def reduction2(C):
    # returns: F - features that must be in the solution
    #          P - pairs which can be removed from the sub-problem
    # First find how many pairs each feature covers
    num_covered = C.sum(axis=0)
    # If we iterate in order we can save some time doing double comparisons
    order = np.argsort(num_covered)
    redundant_features = set()
    for i, f1 in enumerate(order[:-1]):
        for f2 in order[i+1:]:
            # if examples covered by f1 are a subset of those covered by f2
            # then f1 is a redundant feature since f2 covers all its pairs
            if all(np.in1d(np.flatnonzero(C[:, f1]),
                           np.flatnonzero(C[:, f2]))):
            # if np.array_equal(C[:, f1], C[:, f1] * C[:, f2]):
                redundant_features.add(f1)
                break
    return sorted(list(redundant_features))


def reduction3(C):
    # returns: F - features that must be in the solution
    #          P - pairs which can be removed from the sub-problem
    # First find how features each pair is covered by
    num_covered = C.sum(axis=1)
    # If we iterate in order we can save some time doing double comparisons
    order = np.argsort(num_covered)[::-1]
    redundant_pairs = set()
    for i, p1 in enumerate(order[:-1]):
        for p2 in order[i+1:]:
            # if features covering p2 are a subset of those covering p1 then
            # p1 is redundant since any feature that covers p2 will cover p1
            if all(np.in1d(np.flatnonzero(C[p2, :]),
                           np.flatnonzero(C[p1, :]))):
                redundant_pairs.add(p1)
                break
    return sorted(list(redundant_pairs))


def apply(C, apply1=True, apply2=True, apply3=True):
    finished = False
    Np, Nf = C.shape
    P = set(range(Np))
    F = set(range(Nf))
    C_ = C
    forced = set()
    while(not finished):
        finished = True
        if apply1:
            # print('checking 1')
            f, p = reduction1(C_)
            if len(f):
                # print('applying 1')
                finished = False
                # get actual indices
                f = np.array(list(F))[f]
                p = np.array(list(P))[p]
                # update masks
                P.difference_update(p)
                F.difference_update(f)
                # record required features
                forced.update(f)
                C_ = C[list(P), :][:, list(F)]
        if apply2:
            # print('checking 2')
            f = reduction2(C_)
            if len(f):
                # print('applying 2')
                finished = False
                # get actual indices
                f = np.array(list(F))[f]
                F.difference_update(f)
                C_ = C[list(P), :][:, list(F)]
        if apply3:
            # print('checking 3')
            p = reduction3(C_)
            if len(p):
                # print('applying 3')
                finished = False
                # get actual indices
                p = np.array(list(P))[p]
                P.difference_update(p)
                C_ = C[list(P), :][:, list(F)]
        # print({i for i, p in enumerate(Pmask) if p})
    return forced, F, P
