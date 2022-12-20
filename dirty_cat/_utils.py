import collections
from abc import ABC, abstractmethod
from numbers import Integral, Real
from typing import Any, Hashable

import numpy as np
from sklearn.utils import check_array

try:
    # Works for sklearn >= 1.0
    from sklearn.utils import parse_version  # noqa
except ImportError:
    # Works for sklearn < 1.0
    from sklearn.utils.fixes import _parse_version as parse_version  # noqa


class LRUDict:
    """dict with limited capacity

    Using LRU eviction avoids memorizing a full dataset"""

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


def check_input(X) -> np.ndarray:
    """
    Check input with sklearn standards.
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


# Inspired from: https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/utils/_param_validation.py # noqa


def _type_name(t):
    """Convert type into human readable string."""
    module = t.__module__
    qualname = t.__qualname__
    if module == "builtins":
        return qualname
    elif t == Real:
        return "float"
    elif t == Integral:
        return "int"
    return f"{module}.{qualname}"


class _Constraint(ABC):
    """Base class for the constraint objects."""

    def __init__(self):
        self.hidden = False

    @abstractmethod
    def is_satisfied_by(self, val):
        """Whether or not a value satisfies the constraint.
        Parameters
        ----------
        val : object
            The value to check.
        Returns
        -------
        is_satisfied : bool
            Whether or not the constraint is satisfied by this value.
        """

    @abstractmethod
    def __str__(self):
        """A human readable representational string of the constraint."""


class Options(_Constraint):
    """Constraint representing a finite set of instances of a given type.
    Parameters
    ----------
    type : type
    options : set
        The set of valid scalars.
    deprecated : set or None, default=None
        A subset of the `options` to mark as deprecated in the string
        representation of the constraint.
    """

    def __init__(self, type, options, *, deprecated=None):
        super().__init__()
        self.type = type
        self.options = options
        self.deprecated = deprecated or set()

        if self.deprecated - self.options:
            raise ValueError("The deprecated options must be a subset of the options.")

    def is_satisfied_by(self, val):
        return isinstance(val, self.type) and val in self.options

    def _mark_if_deprecated(self, option):
        """Add a deprecated mark to an option if needed."""
        option_str = f"{option!r}"
        if option in self.deprecated:
            option_str = f"{option_str} (deprecated)"
        return option_str

    def __str__(self):
        options_str = (
            f"{', '.join([self._mark_if_deprecated(o) for o in self.options])}"
        )
        return f"a {_type_name(self.type)} among {{{options_str}}}"


class StrOptions(Options):
    """Constraint representing a finite set of strings.

    Parameters
    ----------
    options : set of str
        The set of valid strings.
    deprecated : set of str or None, default=None
        A subset of the `options` to mark as deprecated in the string
        representation of the constraint.
    """

    def __init__(self, options, *, deprecated=None):
        super().__init__(type=str, options=options, deprecated=deprecated)
