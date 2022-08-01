import unittest

import numpy as np

from typing import List, Tuple

try:
    import Levenshtein
except ImportError:
    Levenshtein = False

from dirty_cat import string_distances


def test_get_unique_ngrams() -> None:
    string = 'test'
    true_ngrams = {
        (' ', 't'), ('t', 'e'), ('e', 's'), ('s', 't'),
        ('t', ' '), (' ', 't', 'e'), ('t', 'e', 's'),
        ('e', 's', 't'), ('s', 't', ' '), (' ', 't', 'e', 's'),
        ('t', 'e', 's', 't'), ('e', 's', 't', ' ')
    }
    ngram_range = (2, 4)
    ngrams = string_distances.get_unique_ngrams(string, ngram_range)
    assert ngrams == true_ngrams


def _random_string_pairs(n_pairs=50, seed=1) -> List[Tuple[str, str]]:
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


def _random_common_char_pairs(n_pairs: int = 50, seed: int = 1):
    """
    Return string pairs with a common char at random positions, in order to
    distinguish different thresholds for matching characters in Jaro
    distance.
    """
    # Make strings with random length and common char at index 0
    rng = np.random.RandomState(seed=seed)
    list1 = ['a' + 'b' * rng.randint(2, 20) for k in range(n_pairs)]
    list2 = ['a' + 'c' * rng.randint(2, 20) for k in range(n_pairs)]
    # Shuffle strings
    list1 = [''.join(rng.choice(
        list(s), size=len(s), replace=False)) for s in list1]
    list2 = [''.join(rng.choice(
        list(s), size=len(s), replace=False)) for s in list2]
    pairs = zip(list1, list2)
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


def _check_symmetry(dist_func, *args, **kwargs) -> None:
    for (a, b) in _random_string_pairs():
        assert dist_func(
            a, b, *args, **kwargs) == dist_func(b, a, *args, **kwargs)


def _check_levenshtein_distances() -> None:
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


def test_levenshtein_ratio() -> None:
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
def test_jaro() -> None:
    # If no character in common: similarity is 0
    assert string_distances.jaro('Brian', 'Jesus') == 0
    assert string_distances.jaro_winkler('Brian', 'Jesus') == 0


def test_identical_strings() -> None:
    # Test that if 2 strings are the same, the similarity
    for string1, _ in _random_string_pairs(n_pairs=10):
        assert string_distances.jaro(string1, string1) == 1
        assert string_distances.jaro_winkler(string1, string1) == 1
        assert string_distances.levenshtein_ratio(string1, string1) == 1


def test_compare_implementations() -> None:
    # Compare the implementations of python-Levenshtein to our
    # pure-Python implementations
    if Levenshtein is False:
        raise unittest.SkipTest
    # Test on strings with randomly placed common char
    for string1, string2 in _random_common_char_pairs(n_pairs=50):
        assert (string_distances._jaro_winkler(string1, string2,
                                               winkler=False)
                == Levenshtein.jaro(string1, string2)
                )
        assert (string_distances._jaro_winkler(string1, string2,
                                               winkler=True)
                == Levenshtein.jaro_winkler(string1, string2))
        assert (string_distances.levenshtein_ratio(string1, string2)
                == Levenshtein.ratio(string1, string2))
    # Test on random strings
    for string1, string2 in _random_string_pairs(n_pairs=50):
        assert (string_distances._jaro_winkler(string1, string2,
                                               winkler=False)
                == Levenshtein.jaro(string1, string2))
        assert (string_distances._jaro_winkler(string1, string2,
                                               winkler=True)
                == Levenshtein.jaro_winkler(string1, string2))
        assert (string_distances.levenshtein_ratio(string1, string2)
                == Levenshtein.ratio(string1, string2))


def test_ngram_similarity() -> None:
    # TODO
    # assert ...
    for n in range(1, 4):
        _check_symmetry(string_distances.ngram_similarity, n)
