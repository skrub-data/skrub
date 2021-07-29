import collections

import numpy as np


class LRUDict:
    """ dict with limited capacity

    Using LRU eviction, this avoid to memorizz a full dataset"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def __getitem__(self, key):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return -1

    def __setitem__(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def __contains__(self, key):
        return key in self.cache


def check_input(X):
    """
    Check input data shape.
    Also converts X to a numpy array if not already.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(
            'Expected 2D array. Reshape your data either using'
            'array.reshape(-1, 1) if your data has a single feature or'
            'array.reshape(1, -1) if it contains a single sample.'
        )
    return X
