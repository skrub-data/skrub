import numpy as np

from dirty_cat import string_distances


def _random_string_pairs(n_pairs=50, seed=1):
    rng = np.random.RandomState(seed)
    characters = list(map(chr, range(10000)))
    pairs = []
    for n in range(n_pairs):
        s1_len = rng.randint(50)
        s2_len = rng.randint(50)
        s1 = ''.join(rng.choice(characters, s1_len))
        s2 = ''.join(rng.choice(characters, s2_len))
        pairs.append((s1, s2))
    return pairs


def _check_symmetry(dist_func, *args, **kwargs):
    for (a, b) in _random_string_pairs():
        assert dist_func(
            a, b, *args, **kwargs) == dist_func(b, a, *args, **kwargs)


def test_ngram_similarity():
    # TODO
    # assert ...
    for n in range(1, 4):
        _check_symmetry(string_distances.ngram_similarity, n)
