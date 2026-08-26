import pytest

from skrub._fast_hash import MAXINT32, ngram_min_hash
from skrub.tests.utils import generate_data


def test_fast_hash():
    data = generate_data(100, as_list=True)
    a = data[0]

    min_hash = ngram_min_hash(a, seed=0)
    min_hash2 = ngram_min_hash(a, seed=0)
    assert min_hash == min_hash2

    list_min_hash = [ngram_min_hash(a, seed=seed) for seed in range(50)]
    assert len(set(list_min_hash)) > 45, "Too many hash collisions"

    min_hash4 = ngram_min_hash(a, seed=0, return_minmax=True)
    assert len(min_hash4) == 2


@pytest.mark.parametrize("ngram_range", [(2, 2), (3, 3), (2, 4)])
def test_ngram_range_is_inclusive(ngram_range):
    # Non-regression test: the upper bound of ``ngram_range`` must be
    # inclusive (``min_n <= n <= max_n``). A previous off-by-one excluded
    # ``max_n``, so a single-size range such as ``(3, 3)`` hashed no n-gram
    # at all and returned the ``MAXINT32`` sentinel for every string.
    strings = ["hello", "world", "foobar", "skrubbing"]
    hashes = [ngram_min_hash(s, ngram_range=ngram_range, seed=0) for s in strings]
    assert all(h != MAXINT32 for h in hashes)
    assert len(set(hashes)) == len(strings)


def test_max_ngram_is_used():
    # The ``max_n`` n-gram size must contribute to the hash: across seeds,
    # widening the range from ``(2, 3)`` to ``(2, 4)`` changes the output,
    # confirming 4-grams are incorporated rather than silently dropped.
    string = "skrubbing"
    hashes_without_max = [
        ngram_min_hash(string, ngram_range=(2, 3), seed=seed) for seed in range(50)
    ]
    hashes_with_max = [
        ngram_min_hash(string, ngram_range=(2, 4), seed=seed) for seed in range(50)
    ]
    assert hashes_without_max != hashes_with_max
