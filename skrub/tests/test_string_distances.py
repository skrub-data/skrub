import numpy as np

from skrub import _string_distances


def test_get_unique_ngrams() -> None:
    string = "test"
    true_ngrams = {
        (" ", "t"),
        ("t", "e"),
        ("e", "s"),
        ("s", "t"),
        ("t", " "),
        (" ", "t", "e"),
        ("t", "e", "s"),
        ("e", "s", "t"),
        ("s", "t", " "),
        (" ", "t", "e", "s"),
        ("t", "e", "s", "t"),
        ("e", "s", "t", " "),
    }
    ngram_range = (2, 4)
    ngrams = _string_distances.get_unique_ngrams(string, ngram_range)
    assert ngrams == true_ngrams


def _random_string_pairs(n_pairs=50, seed=1) -> list[tuple[str, str]]:
    rng = np.random.RandomState(seed)
    characters = list(map(chr, range(10000)))
    pairs = []
    for n in range(n_pairs):
        s1_len = rng.randint(50)
        s2_len = rng.randint(50)
        s1 = "".join(rng.choice(characters, s1_len))
        s2 = "".join(rng.choice(characters, s2_len))
        pairs.append((s1, s2))
    return pairs


def _check_symmetry(dist_func, *args, **kwargs) -> None:
    for a, b in _random_string_pairs():
        assert dist_func(a, b, *args, **kwargs) == dist_func(b, a, *args, **kwargs)


def test_ngram_similarity() -> None:
    # TODO
    # assert ...
    for n in range(1, 4):
        _check_symmetry(_string_distances.ngram_similarity, n)
