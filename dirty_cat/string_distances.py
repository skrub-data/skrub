"""
Some string distances
"""
import numpy as np

# Levenstein, adapted from
# https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python

def levenshtein(source, target):
    target_size = len(target)
    if len(source) < target_size:
        return levenshtein(target, source)

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


if __name__ == '__main__':
 print(levenshtein('Varoquaux', 'Gouillart'))
 for i in range(10000):
    levenshtein('Varoquaux', 'Gouillart')



