# from boolnet.learning.feature_selection import abk_file, minimum_feature_set
import minfs.reduction_rules as rr
import datatools.reduction_rules as rr_truth
import bitpacking.packing as pk
import numpy as np
from numpy.testing import assert_array_equal
import pytest


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
    Pmask = np.full(Np, True, dtype=bool)
    Fmask = np.full(Nf, True, dtype=bool)

    forced_, Fmask, Pmask = rr_truth.apply(C, Fmask, Pmask, apply1, apply2, apply3)
    
    forced_ = set(forced_)
    F_ = {i for i, f in enumerate(Fmask) if f}
    P_ = {i for i, p in enumerate(Pmask) if p}
    Csub_ = C[Pmask,:][:,Fmask]

    PFcov, FPcov = rr.build_coverage_maps(X, y)

    forced, F, P = rr.apply_reduction_rules(PFcov, FPcov, apply1, apply2, apply3)
    
    Csub1 = pk.unpackmat(PFcov, Nf, transpose=False)[Pmask, :][:, Fmask]
    Csub2 = pk.unpackmat(FPcov, Np, transpose=True)[Pmask, :][:, Fmask]

    assert forced == forced_
    assert F == F_
    assert P == P_
    assert_array_equal(Csub_, Csub1)
    assert_array_equal(Csub_, Csub2)

