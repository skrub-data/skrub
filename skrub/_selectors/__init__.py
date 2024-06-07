"""
Helpers for selecting columns in a dataframe.
=============================================

A selector represents a column selection rule, such as "all columns that have
numerical data types, except the column 'User ID'". When applied to a dataframe,
it expands into a list of column names that match the selection rule.
This is implemented by the ``Selector`` class and its ``expand`` method.

The 2 advantages offered by selectors are:

- Expressing complex selection rules in a simple and concise way, because they
  can be combined with operators and a range of useful selectors is provided by
  this module.
- Delayed selection: passing a selection rule, to be evaluated later on a
  dataframe that is not yet available. For example, without selectors (i.e. ATM)
  it is not possible to instantiate a skrub.SelectCols that selects "all
  columns that have numerical data types, except the column 'User ID'", if the
  data on which it will be fitted is not yet available.

Here is an example dataframe:

>>> import pandas as pd
>>> df = pd.DataFrame(
...     {
...         "height_mm": [297.0, 420.0],
...         "width_mm": [210.0, 297.0],
...         "kind": ["A4", "A3"],
...         "ID": [4, 3],
...     }
... )

A simple kind of selector selects a fixed list of column names. Such selectors
are created with the ``cols()`` function.

>>> from skrub import _selectors as s
>>> mm_cols = s.cols("height_mm", "width_mm")
>>> mm_cols
cols('height_mm', 'width_mm')
>>> mm_cols.expand(df)
['height_mm', 'width_mm']

Another simple selector, especially useful for default arguments, keeps all
columns:

>>> s.all().expand(df)
['height_mm', 'width_mm', 'kind', 'ID']

Selectors can be combined with operators, for example if we wanted all columns
except the "mm" columns above:

>>> (s.all() - s.cols("height_mm", "width_mm")).expand(df)
['kind', 'ID']

Several kinds of selectors are provided by this module, to select columns by
name, data type, contents or with arbitrary user-provided rules.

>>> s.numeric().expand(df)
['height_mm', 'width_mm', 'ID']
>>> s.glob('*_mm').expand(df)
['height_mm', 'width_mm']

See the full list:

>>> s.ALL_SELECTORS
['Filter', 'NameFilter', 'all', 'any_date', 'boolean', 'cardinality_below', 'categorical', 'cols', 'filter', 'filter_names', 'float', 'glob', 'has_nulls', 'integer', 'inv', 'numeric', 'regex', 'string']

The available operators are |, &, -, ^ with the usual meaning (the same meaning
they would on python sets of the selected column names), and ~ to invert a
selection.

>>> s.glob('*_mm').expand(df)
['height_mm', 'width_mm']
>>> (~s.glob('*_mm')).expand(df)
['kind', 'ID']
>>> (s.glob('*_mm') | s.cols('ID')).expand(df)
['height_mm', 'width_mm', 'ID']
>>> (s.glob('*_mm') & s.glob('height_*')).expand(df)
['height_mm']
>>> (s.glob('*_mm') ^ s.string()).expand(df)
['height_mm', 'width_mm', 'kind']

The operators respect the usual short-circuit rules so for example the
following selector:

>>> s.categorical() & s.cardinality_below(10)
(categorical() & cardinality_below(10))

Will not compute the cardinality of non-categorical columns.

Selecting columns by name
-------------------------

A column name or sequence of column names is converted to a ``cols`` selector
when combined with other selectors in an expression.

>>> s.numeric() - "ID"
(numeric() - cols('ID'))
>>> (s.numeric() - "ID").expand(df)
['height_mm', 'width_mm']

More generally, a column name or list of column names should be accepted by all
skrub public interfaces that accept a selector. This is made easy by the
``make_selector`` function.

>>> s.make_selector(s.all())
all()
>>> s.make_selector("ID")
cols('ID')
>>> s.make_selector(["ID", "kind"])
cols('ID', 'kind')

The cols selector, when used by itself, raises if some of the columns are
missing from the dataframe.

>>> s.cols("ID", "missing").expand(df)
Traceback (most recent call last):
    ...
ValueError: The following columns are requested for selection but missing from dataframe: ['missing']

However, that is no longer the case as soon as it is combined with other selectors.
Indeed, when combining the cols selector with other selectors, we are no longer
performing a simple indexing of the dataframe with a list of columns that we
know exist.

>>> (s.all() & s.cols("ID", "missing")).expand(df)
['ID']

Selectors in skrub interfaces
-----------------------------

In practice, the ``expand`` method of selectors will rarely be called directly by
client code. Rather, users would create selectors and pass them instead of a
column list to skrub objects that operate on a subset of columns. This could
be, for example, the cols parameter of SelectCols: SelectCols(s.glob("*_mm")).

This is not yet the case for SelectCols, as _selectors is still a completely
private module. One example of a (private) function that consumes selectors is
the ``select`` function provided in this module.

>>> s.select(df, ["ID", "kind"])
   ID kind
0   4   A4
1   3   A3
>>> s.select(df, s.numeric() - "ID")
   height_mm  width_mm
0      297.0     210.0
1      420.0     297.0

Advanced selectors: filter and filter_names
-------------------------------------------

``filter`` and ``filter_names`` allow selecting columns based on arbitrary user-defined
criteria. These are also used to implement many of the other selectors provided
in this module.

``filter`` accepts a predicate that is called with a column (pandas or polars
Series) and returns True if it should be selected.

>>> s.filter(lambda col: "A4" in col.tolist()).expand(df)
['kind']

``filter_names`` accepts a predicate that is passed the column name, instead of the
column.

>>> s.filter_names(lambda name: name.endswith('mm')).expand(df)
['height_mm', 'width_mm']

We can pass args and kwargs that will be passed to the predicate, which can
help avoid lambda or local functions and thus ensure the selector is picklable.

>>> s.filter_names(str.endswith, 'mm').expand(df)
['height_mm', 'width_mm']

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

>>> from skrub._selectors._base import NameFilter

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
