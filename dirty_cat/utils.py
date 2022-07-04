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


# def check_input(X):
#     """
#     Check input data shape.
#     Also converts X to a numpy array if not already.
#     """
#     X = np.asarray(X)
#     if X.ndim != 2:
#         raise ValueError(
#             'Expected 2D array. Reshape your data either using'
#             'array.reshape(-1, 1) if your data has a single feature or'
#             'array.reshape(1, -1) if it contains a single sample.'
#         )
#     return X


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


# def check_input(X, y=None):
#     """
#     Check input with sklearn standards.
#     Also converts X to a numpy array if not already.
#     """
#     accepted_types = [np.float64, np.float32, np.float16,
#                       np.int64, np.int32, np.int16,
#                       np.uint64, np.uint32, np.uint16,
#                       np.bool, np.string_, np.datetime64, np.timedelta64, str]
#
#     # Convert date column to utc while keeping the original timestamp
#     date_columns = []
#     timezones = []
#     print(0)
#     print(X)
#     if hasattr(X, "iloc"):  # if X is a pandas.DataFrame
#         for i in range(X.shape[1]):
#             # check if column countains a date
#             if isinstance(X.iloc[0, i], pd.Timestamp):
#                 date_columns.append(i)
#                 timezones.append(X.iloc[0, i].tz)
#                 print(X.iloc[0, i].tz)
#                 if not X.iloc[0, i].tz is None:
#                     X.iloc[:, i] = X.iloc[:, i].dt.tz_convert('UTC')
#     print(1)
#     print(X)
#
#
#     if y is None:
#         X = check_array(X,
#                         dtype=None,
#                         ensure_2d=True,
#                         force_all_finite=False)
#         X = np.asarray(X)
#         # Convert the date columns back to the original timezone
#         for i in date_columns:
#             X[:, i] = pd.to_datetime(X[:, i]).tz_localize(timezones[i])
#         print(2)
#         print(X)
#         print(X.dtype)
#         return X
#
#     else:
#         X, y = check_X_y(X, y,
#                          dtype=accepted_types,
#                          ensure_2d=True,
#                          force_all_finite=False)
#         X = np.asarray(X)
#         for i in date_columns:
#             X[:, i] = pd.to_datetime(X[:, i]).tz_convert(timezones[i])
#         y = np.asarray(y)
#         return X, y

#
# def check_input(X, force_all_finite=False):
#     """
#     Copied from a method used inside sklearn OneHotEncoder.
#     Perform custom check_array:
#     - convert list of strings to object dtype
#     - check for missing values for object dtype data (check_array does
#       not do that)
#     - return list of features (arrays): this list of features is
#       constructed feature by feature to preserve the data types
#       of pandas DataFrame columns, as otherwise information is lost
#       and cannot be used, e.g. for the `categories_` attribute.
#     """
#     accepted_types = [np.float64, np.float32, np.float16,
#                            np.int64, np.int32, np.int16,
#                            np.uint64, np.uint32, np.uint16,
#                            np.bool, np.string_, np.datetime64, np.object]
#     if not (hasattr(X, "iloc") and getattr(X, "ndim", 0) == 2):
#         # if not a dataframe, do normal check_array validation
#         X_temp = check_array(X, dtype=None, force_all_finite=force_all_finite)
#         if not hasattr(X, "dtype") and np.issubdtype(X_temp.dtype, np.str_):
#             X = check_array(X, dtype=object, force_all_finite=force_all_finite)
#         else:
#             X = X_temp
#         needs_validation = False
#     else:
#         # pandas dataframe, do validation later column by column, in order
#         # to keep the dtype information to be used in the encoder.
#         needs_validation = force_all_finite
#
#     n_samples, n_features = X.shape
#     X_columns = []
#
#     for i in range(n_features):
#         if hasattr(X, "iloc"):
#             Xi = X.iloc[:, i]
#         else:
#             Xi = X[:, i]
#         Xi = check_array(
#             Xi, ensure_2d=False, dtype=accepted_types, force_all_finite=needs_validation
#         )
#         X_columns.append(Xi)
#
#
#
#     return np.stack(X_columns, axis=1)#, n_samples, n_features
#
