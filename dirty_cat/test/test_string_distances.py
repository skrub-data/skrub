import unittest

import numpy as np

try:
    import Levenshtein
except ImportError:
    Levenshtein = False

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


def _check_levenshtein_distances():
    for levenshtein_dist in [
            string_distances.levenshtein_seq]:
        _check_levenshtein_example_results(levenshtein_dist)
        _check_symmetry(levenshtein_dist)
    for (a, b) in _random_string_pairs():
        assert string_distances.levenshtein_array(
            a, b) == string_distances.levenshtein_seq(a, b)
        assert string_distances.levenshtein_seq(
            a, b) == string_distances.levenshtein(a, b)
        if Levenshtein is not False:
            assert string_distances.levenshtein_seq(
                a, b) == Levenshtein.distance(a, b)


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


# Tests for jaro
def test_jaro():
    # If no character in common: similarity is 0
    assert string_distances.jaro('Brian', 'Jesus') == 0
    assert string_distances.jaro_winkler('Brian', 'Jesus') == 0


def test_identical_strings():
    # Test that if 2 strings are the same, the similarity
    for string1, _ in _random_string_pairs(n_pairs=10):
        assert string_distances.jaro(string1, string1) == 1
        assert string_distances.jaro_winkler(string1, string1) == 1
        assert string_distances.levenshtein_ratio(string1, string1) == 1


def test_compare_implementations():
    # Compare the implementations of python-Levenshtein to our
    # pure-Python implementations
    if Levenshtein is False:
        raise unittest.SkipTest
    for string1, string2 in _random_string_pairs(n_pairs=10):
        assert (string_distances._jaro_winkler(string1, string2,
                    winkler=False)
                == Levenshtein.jaro(string1, string2)
               )
        assert (string_distances._jaro_winkler(string1, string2)
                == Levenshtein.jaro_winkler(string1, string2)
               )
        assert (string_distances.levenshtein_ratio(string1, string2)
                == Levenshtein.ratio(string1, string2)
               )



def test_ngram_similarity():
    # TODO
    # assert ...
    for n in range(1, 4):
        _check_symmetry(string_distances.ngram_similarity, n)
