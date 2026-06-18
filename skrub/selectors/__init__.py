"""
Helpers for selecting columns in a dataframe.
=============================================
``skrub.selectors`` provides a composable API for selecting columns by
datatype, name, cardinality, null ratio, and more.  Typical usage:

from skrub import selectors as s
from skrub import ApplyToCols, StringEncoder
sel = s.string() | s.categorical()  # all string or categorical columns
# Apply a StringEncoder to all string or categorical columns in a dataframe
ApplyToCols(StringEncoder(), cols=sel).fit_transform(df)

See the User Guide selectors page for the public-facing part of the selector API.

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
