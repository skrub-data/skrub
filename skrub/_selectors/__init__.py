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
...
>>> from skrub import selectors as sbs
>>> sbs.select(df, ["ID", "kind"])
  kind  ID
0   A4   4
1   A3   3
>>> sbs.select(df, sbs.all())
   height_mm  width_mm kind  ID
0      297.0     210.0   A4   4
1      420.0     297.0   A3   3
>>> sbs.select(df, sbs.numeric() - "ID")
   height_mm  width_mm
0      297.0     210.0
1      420.0     297.0
>>> sbs.select(df, sbs.string() | ["width_mm", "ID"])
   width_mm kind  ID
0     210.0   A4   4
1     297.0   A3   3
>>> sbs.select(df, sbs.numeric() - sbs.glob("*_mm"))
   ID
0   4
1   3
>>> s = sbs.numeric() - sbs.glob("*_mm")
>>> s
(numeric() - glob('*_mm'))
>>> sbs.select(df, s)
   ID
0   4
1   3
"""
from ._atoms import created_by, filter, filter_names, glob, regex
from ._base import all, cols, inv, make_selector, nothing, select
from ._dtype_atoms import anydate, boolean, categorical, numeric, string
from ._statistic_atoms import cardinality_below

__all__ = [
    "select",
    "make_selector",
    "all",
    "nothing",
    "cols",
    "inv",
    "glob",
    "regex",
    "filter",
    "filter_names",
    "created_by",
    "numeric",
    "anydate",
    "categorical",
    "string",
    "boolean",
    "cardinality_below",
]
