import pytest
import numpy as np
from numpy.testing import assert_array_equal
import minfs.fs_solver_cplex as fss


def test_single_minfs(instance):
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
