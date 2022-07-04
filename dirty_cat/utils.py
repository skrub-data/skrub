import collections

import numpy as np
import pandas as pd
from sklearn.utils import check_X_y, check_array


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
    Check input with sklearn standards.
    Also converts X to a numpy array if not already.
    """
    #TODO check for weird type of input to pass scikit learn tests
    # whithout messing with the original type too much

    X_ = check_array(X,
                dtype= None,
                ensure_2d=True,
                force_all_finite=False)
    #If the array contains both NaNs and strings, convert to object type
    if X_.dtype.kind in {'U', 'S'}: # contains strings
        if np.any(X_ == "nan"): # missing value converted to string
            return np.array(X, dtype=object)

    return X_