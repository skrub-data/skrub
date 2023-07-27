"""
Some string distances
"""

import re
from collections import Counter

# TODO vectorize these functions (accept arrays)


def get_ngram_count(string: str, ngram_range: tuple[int, int]) -> int:
    """
    Compute the number of ngrams in a string.

    Here is where the formula comes from:

    * the number of 3-grams in a string is the number of sliding windows of
      size 3 in the string: len(string) - 3 + 1
    * this can be generalized to n-grams by changing 3 by n.
    * when given a ngram_range, we can sum this formula over all possible
      ngrams

    """
    min_n, max_n = ngram_range
    ngram_count = 0

    for i in range(min_n, max_n + 1):
        ngram_count += len(string) - i + 1

    return ngram_count


def preprocess(x: str) -> str:
    """
    Combine preprocessing done by CountVectorizer and the SimilarityEncoder.

    Different methods exist to compute the number of ngrams in a string:

    - Simply sum the values of a count vector, which is the output of a
      CountVectorizer with analyzer="char", and a specific ngram_range
    - Compute the number of ngrams using a formula (see ``get_ngram_count``)

    However, in the first case, some preprocessing is done by the
    CountVectorizer that may change the length of the string (in particular,
    stripping sequences of 2 or more whitespaces into 1). In order for the two
    methods to output similar results, this pre-processing is done upstream,
    prior to the CountVectorizer.
    """

    # Preprocessing step done in ngram_similarity
    x = f" {x} "

    # Preprocessing step done in the CountVectorizer
    _white_spaces = re.compile(r"\s\s+")

    return _white_spaces.sub(" ", x)


def get_unique_ngrams(string: str, ngram_range: tuple[int, int]):
    """
    Return the set of unique n-grams of a string.

    Parameters
    ----------
    string : str
        The string to split in n-grams.
    ngram_range : tuple (min_n, max_n)
        The lower and upper boundaries of the range of n-values for different
        n-grams used in the string similarity. All values of `n` such
        that ``min_n <= n <= max_n`` will be used.

    Returns
    -------
    set
        The set of unique n-grams of the string.
    """
    spaces = " "  # * (n // 2 + n % 2)
    string = spaces + " ".join(string.lower().split()) + spaces
    ngram_set = set()
    for n in range(ngram_range[0], ngram_range[1] + 1):
        string_list = [string[i:] for i in range(n)]
        ngram_set |= set(zip(*string_list))
    return ngram_set


def get_ngrams(string: str, n: int) -> list[tuple]:
    """Return the set of different n-grams in a string"""
    # Pure Python implementation: no numpy
    spaces = " "  # * (n // 2 + n % 2)
    string = spaces + " ".join(string.lower().split()) + spaces
    string_list = [string[i:] for i in range(n)]
    return list(zip(*string_list))


def ngram_similarity(string1, string2, n, preprocess_strings=True):
    """n-gram similarity between two strings"""
    if preprocess_strings:
        string1, string2 = preprocess(string1), preprocess(string2)

    ngrams1 = get_ngrams(string1, n)
    count1 = Counter(ngrams1)

    ngrams2 = get_ngrams(string2, n)
    count2 = Counter(ngrams2)

    samegrams = sum((count1 & count2).values())
    allgrams = len(ngrams1) + len(ngrams2)
    similarity = samegrams / (allgrams - samegrams)
    return similarity
