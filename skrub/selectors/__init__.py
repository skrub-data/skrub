"""
Helpers for selecting columns in a dataframe.
=============================================

See the User Guide selectors page for the public-facing part of the selector API.

Defining new selectors
----------------------

This last advanced section is aimed at skrub developers adding new selectors to
this module.

A Selector subclass must define the ``_matches`` method. It accepts a column and
returns True if the column should be selected.

Additionally, the subclass can override the ``expand`` method. It accepts a
dataframe and returns the list of column names that should be selected. This is
only called when the selector is used by itself. Whenever it is combined with
other selectors with operators, ``_matches`` is used. Overriding ``expand`` thus
allows special-casing the behavior when it is used on its own, such as raising
an exception when a simple list of column names is used for selection and some
are missing from the dataframe. Overriding ``expand`` is not necessary in most
cases; it may actually never be necessary except for the ``cols`` special case.

A simpler alternative to defining a new Selector subclass is to define a
function that constructs a selector by calling ``filter`` or ``filter_names`` with an
appropriate predicate and arguments; most selectors offered by this module are
implemented with this approach.

>>> from skrub import _dataframe as sbd
>>> from skrub import selectors as s
>>> import pandas as pd
>>> df = pd.DataFrame(
...     {
...         "height_mm": [297.0, 420.0],
...         "width_mm": [210.0, 297.0],
...         "kind": ["A4", "A3"],
...         "ID": [4, 3],
...     }
... )

Defining a new class:

>>> class EndsWith(s.Selector):
...     def __init__(self, suffix):
...         self.suffix = suffix
...
...     def _matches(self, col):
...         return sbd.name(col).endswith(self.suffix)

>>> EndsWith('_mm').expand(df)
['height_mm', 'width_mm']

Using a filter:

>>> def ends_with(suffix):
...     return s.filter_names(str.endswith, suffix)

>>> ends_with('_mm').expand(df)
['height_mm', 'width_mm']

>>> ends_with('_mm')
filter_names(str.endswith, '_mm')

Directly instantiating a Filter or FilterNames object allows passing the name
argument and thus controlling the repr of the resulting selector, so an
slightly improved version could be:

>>> from skrub.selectors._base import NameFilter

>>> def ends_with(suffix):
...     return NameFilter(str.endswith, args=(suffix,), name='ends_with')

>>> ends_with('_mm')
ends_with('_mm')

>>> ends_with('_mm').expand(df)
['height_mm', 'width_mm']
"""  # noqa: E501

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
