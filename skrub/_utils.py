import collections
import importlib
import secrets
from typing import Iterable

import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.utils import check_array
from sklearn.utils.fixes import parse_version

from skrub import _dataframe as sbd


class LRUDict:
    """Dict with limited capacity.

    Using LRU eviction avoids memorizing a full dataset.
    """

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


def unique_strings(values, is_null):
    """Unique values, accounting for nulls.

    This is like np.unique except
    - it is only for 1d arrays of strings
    - caller must pass a boolean array indicating which values are null: ``is_null``
    - null values are considered to be the same as the empty string.

    >>> import numpy as np
    >>> import pandas as pd
    >>> from skrub._utils import unique_strings
    >>> a = np.asarray(['paris', '', 'berlin', None, 'london'])
    >>> values, idx = unique_strings(a, pd.isna(a))
    >>> values
    array(['', 'berlin', 'london', 'paris'], dtype=object)
    >>> values[idx]
    array(['paris', '', 'berlin', '', 'london'], dtype=object)
    """
    not_null_values = values[~is_null]
    unique, idx = np.unique(not_null_values, return_inverse=True)
    if not is_null.any():
        return unique, idx
    if not len(unique) or unique[0] != "":
        unique = np.concatenate([[""], unique])
        idx += 1
    full_idx = np.empty(values.shape, dtype=idx.dtype)
    full_idx[is_null] = 0
    full_idx[~is_null] = idx
    return unique, full_idx


def check_input(X):
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


def import_optional_dependency(name, extra=""):
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


def set_output(transformer, X):
    if not hasattr(transformer, "set_output"):
        return
    module_name = sbd.dataframe_module_name(X)
    target_module = module_name
    if module_name == "polars" and parse_version(sklearn.__version__) < parse_version(
        "1.4"
    ):
        # TODO: remove when scikit-learn 1.3 support is dropped.
        target_module = module_name
    try:
        transformer.set_output(transform=target_module)
    except Exception:
        # Some scikit-learn estimators have a set_output method, but it can
        # fail -- for example a Pipeline containing a step that doesn't have
        # set_output. The pipeline may still produce the right output type if
        # it does it by default, without the set_output call. So we allow
        # set_output to fail and attempt the transform, and an error is raised
        # if the output of transform has the wrong type.
        pass


def check_output(
    transformer, transform_input, transform_output, allow_column_list=True
):
    target_module = sbd.dataframe_module_name(transform_input)

    def has_correct_module(obj):
        return sbd.dataframe_module_name(obj) == target_module

    if (
        sbd.is_dataframe(transform_output)
        and target_module == "polars"
        and sbd.dataframe_module_name(transform_output) == "pandas"
        and hasattr(transformer, "set_output")
        and parse_version(sklearn.__version__) < parse_version("1.4")
    ):
        # TODO: remove when scikit-learn 1.3 support is dropped.
        #
        # For older scikit-learn versions that do not support
        # `set_output(transform='polars')`, we fall back to using
        # `set_output(transform='pandas')` and converting the output dataframe
        # to polars ourselves.
        # Therefore having pandas output when the input is polars is tolerated,
        # when:
        #   - the scikit-learn version is < 1.4
        #   - and the transformer relies on the set_output API
        #     (this implies that the output is a dataframe -- not a column or
        #     list of columns).
        # In all other cases having the output backed by the wrong dataframe
        # library (e.g. pandas instead of polars) will result in an error.

        # transform_input is a polars object so we know we can import it
        import polars as pl

        return pl.from_pandas(transform_output)
    if sbd.is_dataframe(transform_output) and has_correct_module(transform_output):
        return transform_output
    if (
        allow_column_list
        and sbd.is_column(transform_output)
        and has_correct_module(transform_output)
    ):
        return transform_output
    if (
        allow_column_list
        and sbd.is_column_list(transform_output)
        and (not len(transform_output) or has_correct_module(transform_output[0]))
    ):
        return transform_output
    message = (
        f"{transformer.__class__.__name__}.fit_transform returned a result of type"
        f" {transform_output.__class__.__name__}, but a {target_module} DataFrame was"
        f" expected. If {transformer.__class__.__name__} is a custom transformer class,"
        f" please make sure that the output is a {target_module} container when the"
        f" input is a {target_module} container."
    )
    if not hasattr(transformer, "set_output"):
        message += (
            f" One way of enabling a transformer to output {target_module} DataFrames"
            " is inheriting from the sklearn.base.TransformerMixin class and defining"
            " the 'get_feature_names_out' method. See"
            " https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_set_output.html"
            " for details."
        )
    raise TypeError(message)
