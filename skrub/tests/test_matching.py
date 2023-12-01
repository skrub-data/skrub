import numpy as np
import pytest

from skrub import _matching


@pytest.mark.parametrize(
    "strategy,expected",
    [
        (_matching.Matching, 2.0),
        (_matching.OtherNeighbor, 0.5),
        (_matching.SelfJoinNeighbor, 1.0),
        (_matching.RandomPairs, 1.0),
    ],
)
def test_matching_rescaled_distance(strategy, expected):
    main = np.asarray([[0.0]])
    aux = np.asarray([2.0, 4.0])[:, None]
    match_result = strategy().fit(aux).match(main, np.inf)
    assert match_result["rescaled_distance"][0] == expected


def test_percentile_bad_params():
    with pytest.raises(ValueError, match="must be a positive integer"):
        _matching.RandomPairs(n_sampled_pairs=0).fit(np.zeros((3, 1)))
    with pytest.raises(ValueError, match="with only 1 rows"):
        _matching.RandomPairs().fit(np.zeros((1, 1)))
