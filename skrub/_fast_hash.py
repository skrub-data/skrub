"""
n-gram hashing by simple dot products

The principle is as follows:
  1. A string is viewed as a succession of numbers (the ASCII or UTF8
     representation of its elements).
  2. Each n-gram is then an n-dimensional vector of integers "g". A simple
     hash function is then computed by taking the dot product with a
     given random vector "atom", modulo max-int (integers larger than
     max-int overflow). The corresponding operation defines a random
     order in the interval [-maxint, maxint]
  3. Computing this dot product over a sliding window (to compute it for
     every n-gram is a convolution)
  4. We can then take the min (or the max) of the resulting sliding window
"""

import functools

import numpy as np

# Precompute to avoid the cost and
# cast to int32 to speed up the min
MININT32 = np.int32(-(2 ** (32 - 1)))
MAXINT32 = np.int32(2 ** (32 - 1) - 1)


@functools.lru_cache(maxsize=1024)
def gen_atom(atom_len, seed=0):
    """
    Generate a random integer array (atom).

    Parameters
    ----------
    atom_len : int
        The length of the atom.
    seed : int, default=0
        The seed of the random_number generator.

    Returns
    -------
    array, shape (atom_len, )
        An array of random integers of length atom_len and dtype int32
        (assuming dtype_size=32).
    """
    rng = np.random.RandomState(seed)
    atom = rng.randint(-MAXINT32, MAXINT32, size=atom_len, dtype=np.dtype("int32"))
    return atom


def ngram_min_hash(
    string: str,
    ngram_range: tuple[int, int] = (2, 4),
    seed: int = 0,
    return_minmax=False,
) -> int | tuple[int, int]:
    """
    Compute the min/max hash of the ngrams of the string.

    Parameters
    ----------
    string : str
        String to encode.
    ngram_range : 2-tuple of int, default=(2, 4)
        The lower and upper boundaries of the range of n-values for different
        n-grams used in the string similarity. All values of `n` such
        that ``min_n <= n <= max_n`` will be used.
    seed : int, default=0
        Integer used to seed the hashing function.
    return_minmax : bool, default=False
        If True, returns both the minhash and maxhash of the string.
        Else, only returns the minhash.

    Returns
    -------
    int or tuple
        The min_hash or (min_hash, max_hash) of the n-grams of the string.
    """
    # Create a numerical 1D array from the string
    array = np.frombuffer(string.encode(), dtype="int8", count=len(string))

    max_hash = MININT32
    min_hash = MAXINT32
    for atom_len in range(ngram_range[0], ngram_range[1]):
        atom = gen_atom(atom_len, seed=seed)
        # np.correlate is faster than np.convolve
        # the convolution gives a hash for each ngram
        hashes = np.correlate(array, atom)
        min_hash = min(min_hash, hashes.min())
        if return_minmax:
            max_hash = max(max_hash, hashes.max())

    # We should check that longer windows do not have different
    # statistics from shorter ones
    if return_minmax:
        return min_hash, max_hash
    return min_hash
