import numpy as np

from dirty_cat import string_distances


def _random_string_pairs(n_pairs=50):
    rng = np.random.RandomState(0)
    characters = list(map(chr, range(10000)))
    pairs = []
    for n in range(n_pairs):
        s1_len = rng.randint(50)
        s2_len = rng.randint(50)
        s1 = ''.join(np.random.choice(characters, s1_len))
        s2 = ''.join(np.random.choice(characters, s2_len))
        pairs.append((s1, s2))
    return pairs


# TODO: some factorization of what is common for distances;
# check results for same examples on all distances
def _check_levenshtein_example_results(levenshtein_dist):
    assert levenshtein_dist('', '') == 0
    assert levenshtein_dist('', 'abc') == 3
    assert levenshtein_dist('abc', '') == 3
    assert levenshtein_dist('abc', 'abc') == 0
    assert levenshtein_dist('abcd', 'abc') == 1
    assert levenshtein_dist('abc', 'abcd') == 1
    assert levenshtein_dist('xbcd', 'abcd') == 1
    assert levenshtein_dist('axcd', 'abcd') == 1
    assert levenshtein_dist('abxd', 'abcd') == 1
    assert levenshtein_dist('abcx', 'abcd') == 1
    assert levenshtein_dist('axcx', 'abcd') == 2
    assert levenshtein_dist('axcx', 'abcde') == 3


def _check_symmetry(dist_func, *args, **kwargs):
    for (a, b) in _random_string_pairs():
        assert dist_func(
            a, b, *args, **kwargs) == dist_func(b, a, *args, **kwargs)


def test_levenshtein_distances():
    for levenshtein_dist in [
            string_distances.levenshtein_seq]:
        _check_levenshtein_example_results(levenshtein_dist)
        _check_symmetry(levenshtein_dist)
    for (a, b) in _random_string_pairs():
        assert string_distances.levenshtein_array(
            a, b) == string_distances.levenshtein_seq(a, b)
        assert string_distances.levenshtein_seq(
            a, b) == string_distances.levenshtein(a, b)


def test_levenshtein_ratio():
    # TODO
    # assert string_distances.levenshtein_ratio('', '') == 1
    # assert string_distances.levenshtein_ratio('', 'abc') == 3
    # assert string_distances.levenshtein_ratio('abc', '') == 3
    # assert string_distances.levenshtein_ratio('abc', 'abc') == 0
    # assert string_distances.levenshtein_ratio('abcd', 'abc') == 1
    # assert string_distances.levenshtein_ratio('abc', 'abcd') == 1
    # assert string_distances.levenshtein_ratio('xbcd', 'abcd') == 1
    # assert string_distances.levenshtein_ratio('axcd', 'abcd') == 1
    # assert string_distances.levenshtein_ratio('abxd', 'abcd') == 1
    # assert string_distances.levenshtein_ratio('abcx', 'abcd') == 1
    # assert string_distances.levenshtein_ratio('axcx', 'abcd') == 2
    # assert string_distances.levenshtein_ratio('axcx', 'abcde') == 3
    pass


def test_ngram_similarity():
    # TODO
    # assert ...
    for n in range(1, 4):
        _check_symmetry(string_distances.ngram_similarity, n)
