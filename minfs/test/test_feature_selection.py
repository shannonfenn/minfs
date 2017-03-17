import minfs.feature_selection as fss
from numpy.testing import assert_array_equal


def test_order_to_ranking_with_ties():
    scores = [3, 6, 3, 2, 1, 9, 9, 3, 2]
    order = [4, 3, 8, 0, 2, 7, 1, 5, 6]
    expected_ranking = [3, 6, 3, 1, 0, 7, 7, 3, 1]
    actual_ranking = fss.order_to_ranking_with_ties(
        order, lambda i, j: scores[i] == scores[j])
    assert_array_equal(expected_ranking, actual_ranking)
