"""
Helpers for selecting columns in a dataframe.
=============================================

``skrub.selectors`` provides a composable API for selecting columns by
datatype, name, cardinality, null ratio, and more.  Typical usage:

>>> from skrub import selectors as s
>>> sel = s.numeric() | s.boolean()  # all numeric or boolean columns
>>> sel(df)  # returns the matching subset of `df`
"""

from . import _selectors
from ._base import (
    Filter,
    NameFilter,
    Selector,
    all,
    cols,
    drop,
    filter,
    filter_names,
    inv,
    make_selector,
    select,
)
from ._selectors import *  # noqa: F403

__all__ = [
    "Filter",
    "NameFilter",
    "Selector",
    "all",
    "cols",
    "filter",
    "filter_names",
    "inv",
    "make_selector",
    "select",
    "drop",
]
__all__ += _selectors.__all__

ALL_SELECTORS = sorted(set(__all__) - {"Selector", "make_selector", "select"})
