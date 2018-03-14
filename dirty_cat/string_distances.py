"""
Some string distances
"""
import numpy as np

# Levenstein, adapted from
# https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python


def levenshtein_array(source, target):
    target_size = len(target)
    if len(source) < target_size:
        return levenshtein_array(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # Create numpy arrays
    source = np.array(tuple(source), dtype='|S1')
    target = np.array(tuple(target), dtype='|S1')

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
    if len(seq1) < 15:
        return levenshtein_seq(seq1, seq2)
    else:
        return levenshtein_array(seq1, seq2)


def get_unique_ngrams(string, n):
    """ Return the set of different tri-grams in a string
    """
    spaces = ' ' * (n // 2 + n % 2)
    string = spaces + string + '  '
    string_list = [string[i:] for i in range(n)]
    return set(zip(*string_list))


def get_ngrams(string, n):
    """ Return the set of different tri-grams in a string
    """
    # Pure Python implementation: no numpy
    spaces = ' ' * (n // 2 + n % 2)
    string = spaces + string + spaces
    string_list = [string[i:] for i in range(n)]
    return list(zip(*string_list))


def ngram_similarity(string1, string2, n):
    """ n-gram similarity between two strings
    """
    ngrams1 = get_ngrams(string1, n)
    count_dict1 = {}
    for ngram in ngrams1:
        count_dict1.setdefault(ngram, 0)
        count_dict1[ngram] += 1

    ngrams2 = get_ngrams(string2, n)
    samegrams = 0
    for ngram in ngrams2:
        try:
            if count_dict1[ngram] > 0:
                count_dict1[ngram] -= 1
                samegrams += 1
        except KeyError:
            continue
    allgrams = len(ngrams1) + len(ngrams2)
    similarity = samegrams/(allgrams - samegrams)
    return similarity


if __name__ == '__main__':
    s1 = 'aa'
    s2 = 'aaa'
    print('Levenshtein similarity: %.3f' % levenshtein(s1, s2))
    print('3-gram similarity: %.3f' % ngram_similarity(s1, s2, 3))
