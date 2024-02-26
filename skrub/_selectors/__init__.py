"""Helpers for selecting columns in a dataframe.

TODO

>>> import pandas as pd
>>> df = pd.DataFrame(
...     {
...         "height_mm": [297.0, 420.0],
...         "width_mm": [210.0, 297.0],
...         "kind": ["A4", "A3"],
...         "ID": [4, 3],
...     }
... )
>>> from skrub import selectors as s
>>> s.select(df, ["ID", "kind"])
  kind  ID
0   A4   4
1   A3   3
>>> s.select(df, s.all())
   height_mm  width_mm kind  ID
0      297.0     210.0   A4   4
1      420.0     297.0   A3   3
>>> s.select(df, s.numeric() - "ID")
   height_mm  width_mm
0      297.0     210.0
1      420.0     297.0
>>> s.select(df, s.string() | ["width_mm", "ID"])
   width_mm kind  ID
0     210.0   A4   4
1     297.0   A3   3
>>> s.select(df, s.numeric() - s.glob("*_mm"))
   ID
0   4
1   3
>>> non_mm = s.numeric() - s.glob("*_mm")
>>> non_mm
(numeric() - glob('*_mm'))
>>> s.select(df, non_mm)
   ID
0   4
1   3
"""

from ._atoms import created_by, filter, filter_names, glob, regex
from ._base import all, cols, inv, make_selector, name_in, nothing, select
from ._dtype_atoms import any_date, boolean, categorical, numeric, string
from ._statistic_atoms import cardinality_below

__all__ = [
    "select",
    "make_selector",
    "all",
    "nothing",
    "cols",
    "name_in",
    "inv",
    "glob",
    "regex",
    "filter",
    "filter_names",
    "created_by",
    "numeric",
    "any_date",
    "categorical",
    "string",
    "boolean",
    "cardinality_below",
]
