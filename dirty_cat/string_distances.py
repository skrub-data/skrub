"""
Some string distances
"""
import functools
import re

import numpy as np

try:
    import Levenshtein
    _LEVENSHTEIN_AVAILABLE = True
except ImportError:
    _LEVENSHTEIN_AVAILABLE = False

from collections import Counter
# Levenstein, adapted from
# https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python


# TODO vectorize these functions (accept arrays)


def get_ngram_count(X, ngram_range):
    """Compute the number of ngrams in a string.

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
        ngram_count += len(X) - i + 1

    return ngram_count


def preprocess(x):
    """Combine preprocessing done by CountVectorizer and the SimilarityEncoder.

    Different methods exist to compute the number of ngrams in a string:

    - Simply sum the values of a count vector, which is the ouput of a
      CountVectorizer with analyzer="char", and a specific ngram_range
    - Compute the number of ngrams using a formula (see ``get_ngram_count``)

    However, in the first case, some preprocessing is done by the
    CountVectorizer that may change the length of the string (in particular,
    stripping sequences of 2 or more whitespaces into 1). In order for the two
    methods to output similar results, this pre-processing is done upstream,
    prior to the CountVectorizer.
    """

    # preprocessing step done in ngram_similarity
    x = ' %s ' % x

    # preprocessing step done in the CountVectorizer
    _white_spaces = re.compile(r"\s\s+")

    return _white_spaces.sub(' ', x)


def levenshtein_array(source, target):
    target_size = len(target)
    if len(source) < target_size:
        return levenshtein_array(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # Create numpy arrays
    source = np.array(tuple(source), dtype='|U1')
    target = np.array(tuple(target), dtype='|U1')

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target_size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]


def levenshtein_seq(seq1, seq2):
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    len_seq2 = len(seq2)
    for x in range(len(seq1)):
        oneago = thisrow
        thisrow = [0] * len_seq2 + [x + 1]
        for y in range(len_seq2):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
    return thisrow[len_seq2 - 1]


def levenshtein(seq1, seq2):
    # Choose the fastest option depending on the size of the arrays
    # The number 15 was chosen empirically on Python 3.6
    if _LEVENSHTEIN_AVAILABLE:
        return Levenshtein.distance(seq1, seq2)
    if len(seq1) < 15:
        return levenshtein_seq(seq1, seq2)
    else:
        return levenshtein_array(seq1, seq2)


def _levenshtein_ratio(seq1, seq2):
    # Private function not using the Levenshtein package
    total_len = len(seq1) + len(seq2)
    if total_len == 0:
        return 1.
    return (total_len - levenshtein(seq1, seq2)) / total_len


if _LEVENSHTEIN_AVAILABLE:
    levenshtein_ratio = Levenshtein.ratio
else:
    levenshtein_ratio = _levenshtein_ratio


def _jaro_winkler(seq1, seq2, winkler=False):
    # Adapted from the jellyfish package
    # If winkler is False, the Jaro similarity is returned, else the
    # Jaro-Winkler variant is returned

    seq1_len = len(seq1)
    seq2_len = len(seq2)

    if not seq1_len or not seq2_len:
        return 0.0

    min_len = max(seq1_len, seq2_len)
    search_range = (min_len // 2) - 1
    if search_range < 0:
        search_range = 0

    seq1_flags = seq1_len * [False]
    seq2_flags = seq2_len * [False]

    # looking only within search range, count & flag matched pairs
    common_chars = 0
    for i, seq1_ch in enumerate(seq1):
        low = i - search_range if i > search_range else 0
        hi = i + search_range if i + search_range < seq2_len else seq2_len - 1
        for j in range(low, hi + 1):
            if not seq2_flags[j] and seq2[j] == seq1_ch:
                seq1_flags[i] = seq2_flags[j] = True
                common_chars += 1
                break

    # short circuit if no characters match
    if not common_chars:
        return 0.0

    # count transpositions
    k = trans_count = 0
    for i, seq1_f in enumerate(seq1_flags):
        if seq1_f:
            for j in range(k, seq2_len):
                if seq2_flags[j]:
                    k = j + 1
                    break
            if seq1[i] != seq2[j]:
                trans_count += 1
    trans_count /= 2

    # adjust for similarities in nonmatched characters
    common_chars = float(common_chars)
    weight = ((common_chars/seq1_len + common_chars/seq2_len +
              (common_chars-trans_count) / common_chars)) / 3

    # winkler modification: continue to boost if strings are similar
    if winkler and weight > 0.7 and seq1_len > 3 and seq2_len > 3:
        # adjust for up to first 4 chars in common
        j = min(min_len, 4)
        i = 0
        while i < j and seq1[i] == seq2[i] and seq1[i]:
            i += 1
        if i:
            weight += i * 0.1 * (1.0 - weight)

    return weight


if _LEVENSHTEIN_AVAILABLE:
    jaro_winkler = Levenshtein.jaro_winkler
    jaro = Levenshtein.jaro
else:
    jaro_winkler = functools.partial(_jaro_winkler, winkler=True)
    jaro = _jaro_winkler


def get_unique_ngrams(string, n):
    """ Return the set of different tri-grams in a string
    """
    spaces = ' '  # * (n // 2 + n % 2)
    string = spaces + " ".join(string.lower().split()) + spaces
    string_list = [string[i:] for i in range(n)]
    return set(zip(*string_list))


def get_ngrams(string, n):
    """ Return the set of different tri-grams in a string
    """
    # Pure Python implementation: no numpy
    spaces = ' '  # * (n // 2 + n % 2)
    string = spaces + " ".join(string.lower().split()) + spaces
    string_list = [string[i:] for i in range(n)]
    return list(zip(*string_list))


def ngram_similarity(string1, string2, n, preprocess_strings=True):
    """ n-gram similarity between two strings
    """
    if preprocess_strings:
        string1, string2 = preprocess(string1), preprocess(string2)

    ngrams1 = get_ngrams(string1, n)
    count1 = Counter(ngrams1)

    ngrams2 = get_ngrams(string2, n)
    count2 = Counter(ngrams2)

    samegrams = sum((count1 & count2).values())
    allgrams = len(ngrams1) + len(ngrams2)
    similarity = samegrams/(allgrams - samegrams)
    return similarity


if __name__ == '__main__':
    s1 = 'aa'
    s2 = 'aaab'
    print('Levenshtein similarity: %.3f' % levenshtein(s1, s2))
    print('3-gram similarity: %.3f' % ngram_similarity(s1, s2, 3))
