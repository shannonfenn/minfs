import minfs.fs_solver_raps as fss
from minfs.feature_selection import is_valid


def test_single_minfs(instance):
    features, target, _ = instance
    print(features.shape, target.shape)
    actual = fss.single_minfs(features, target)
    # check the expected minfs is one of the returned
    assert is_valid(features[:, actual], target)
