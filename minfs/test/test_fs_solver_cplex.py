# from boolnet.learning.feature_selection import abk_file, minimum_feature_set
import minfs.fs_solver_cplex as fss
import numpy as np
from numpy.testing import assert_array_equal
import pytest


def test_single_minfs(instance):
# def tmpfilename():
#     random_suffix = ''.join(str(i) for i in np.random.randint(0, 9, 10))
#     return '/tmp/shantemp' + random_suffix


@pytest.fixture(params=[1, 2, 3, 4, 5])
def instance(request):
    filename_base = 'minfs/test/instances/{}'.format(request.param)
    with np.load(filename_base + '.npz') as data:
        features = data['features']
        target = data['target']
        feature_sets = data['feature_sets']
    return features, target, feature_sets


# def test_abk_file_generation(instance, tmpfilename):
#     features, target, _, abk_file_name = instance
#     abk_file(features, target, tmpfilename)
#     with open(tmpfilename) as f:
#         actual = f.read()
#     with open(abk_file_name) as f:
#         expected = f.read()
#     assert expected == actual


    features, target, all_expected = instance
    actual = fss.single_minfs(features, target)
    # check the expected minfs is one of the returned
    assert any(np.array_equal(actual, expected) for expected in all_expected)


@pytest.mark.skip
def test_all_minfs(instance):
    features, target, expected = instance

    actual = fss.all_minfs(features, target)

    # check the expected minfs is one of the returned
    assert_array_equal(sorted(expected.tolist()),
                       sorted(actual.tolist()))
