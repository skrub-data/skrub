"""
Contains method to select columns in a dataframe.

See the :ref:`selectors <selectors>` section for further details.
"""

from . import _selectors
from ._base import (
    Filter,
    NameFilter,
    Selector,
    all,
    cols,
    filter,
    filter_names,
    inv,
    make_selector,
    select,
)
from ._selectors import *  # noqa: F403,F401

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
]
__all__ += _selectors.__all__

ALL_SELECTORS = sorted(set(__all__) - {"Selector", "make_selector", "select"})
