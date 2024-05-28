import collections
import importlib
import secrets
from collections.abc import Hashable
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray
from sklearn.base import clone
from sklearn.utils import check_array

from skrub import _dataframe as sbd


class LRUDict:
    """Dict with limited capacity.

    Using LRU eviction avoids memorizing a full dataset.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def __getitem__(self, key: Hashable):
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return -1

    def __setitem__(self, key: Hashable, value: Any):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

    def __contains__(self, key: Hashable):
        return key in self.cache


def check_input(X) -> NDArray:
    """Check input with sklearn standards.

    Also converts X to a numpy array if not already.
    """
    # TODO check for weird type of input to pass scikit learn tests
    #  without messing with the original type too much

    X_ = check_array(
        X,
        dtype=None,
        ensure_2d=True,
        force_all_finite=False,
    )
    # If the array contains both NaNs and strings, convert to object type
    if X_.dtype.kind in {"U", "S"}:  # contains strings
        if np.any(X_ == "nan"):  # missing value converted to string
            return check_array(
                np.array(X, dtype=object),
                dtype=None,
                ensure_2d=True,
                force_all_finite=False,
            )

    return X_


def import_optional_dependency(name: str, extra: str = ""):
    """Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice
    message will be raised.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.

    Returns
    -------
    maybe_module : Optional[ModuleType]
        The imported module when found.
    """

    msg = (
        f"Missing optional dependency '{name}'. {extra} "
        f"Use pip or conda to install {name}."
    )
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(msg) from exc

    return module


def atleast_1d_or_none(x):
    """``np.atleast_1d`` helper returning an empty list when x is None."""
    if x is None:
        return []
    return np.atleast_1d(x).tolist()


def _is_array_like(x):
    return (
        isinstance(x, Iterable)
        and not isinstance(x, (str, bytes))
        and not hasattr(x, "__dataframe__")
    )


def atleast_2d_or_none(x):
    """``np.atleast_2d`` helper returning an empty list when x is None.

    Note that we don't use ``np.atleast_2d`` because x could be a jagged array.

    Returns
    -------
    list of lists
        The processed array in 2d shape.
    """
    if x is None:
        return [[]]
    if _is_array_like(x) is not True:
        x = [x]

    is_array_list = [_is_array_like(item) for item in x]

    # 2d array
    if all(is_array_list):
        return [atleast_1d_or_none(item) for item in x]

    # 1d array, mix of scalar and arrays
    elif any(is_array_list):
        raise ValueError(
            f"Mix of array and scalar or string values not accepted, got {x=!r}"
        )

    # 1d array
    else:
        return [x]


def clone_if_default(estimator, default_estimator):
    return clone(estimator) if estimator is default_estimator else estimator


def random_string():
    return secrets.token_hex()[:8]


def get_duplicates(values):
    counts = collections.Counter(values)
    duplicates = [k for k, v in counts.items() if v > 1]
    return duplicates


def check_duplicated_column_names(column_names, table_name=None):
    duplicates = get_duplicates(column_names)
    if duplicates:
        table_name = "" if table_name is None else f"{table_name!r}"
        raise ValueError(
            f"Table {table_name} has duplicate column names: {duplicates}."
            " Please make sure column names are unique."
        )


def renaming_func(renaming):
    if isinstance(renaming, str):
        return renaming.format
    return renaming


def repr_args(args, kwargs, defaults={}):
    return ", ".join(
        [repr(a) for a in args]
        + [
            f"{k}={v!r}"
            for k, v in kwargs.items()
            if k not in defaults or defaults[k] != v
        ]
    )


def transformer_output_type_error(transformer, transform_input, transform_output):
    module = sbd.dataframe_module_name(transform_input)
    message = (
        f"{transformer.__class__.__name__}.fit_transform returned a result of type"
        f" {transform_output.__class__.__name__}, but a {module} DataFrame was"
        f" expected. If {transformer.__class__.__name__} is a custom transformer class,"
        f" please make sure that the output is a {module} container when the input is a"
        f" {module} container."
    )
    if not hasattr(transformer, "set_output"):
        message += (
            f" One way of enabling a transformer to output {module} DataFrames is"
            " inheriting from the sklearn.base.TransformerMixin class and defining the"
            " 'get_feature_names_out' method. See"
            " https://scikit-learn.org/stable/auto_examples/"
            "miscellaneous/plot_set_output.html"
            " for details."
        )
    raise TypeError(message)
