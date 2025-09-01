Usage
-----

Here is an example dataframe. Note that selectors support both Pandas and Polars
dataframes:

>>> import pandas as pd
>>> df = pd.DataFrame(
...     {
...         "height_mm": [297.0, 420.0],
...         "width_mm": [210.0, 297.0],
...         "kind": ["A4", "A3"],
...         "ID": [4, 3],
...     }
... )

:func:`~skrub.selectors.cols` is a simple kind of selector which selects a fixed list of
column names:

>>> from skrub import selectors as s
>>> mm_cols = s.cols('height_mm', 'width_mm')
>>> mm_cols
cols('height_mm', 'width_mm')

This selector can then be passed to a :func:`~skrub.selectors.select` function:

>>> s.select(df, mm_cols)
   height_mm  width_mm
0      297.0     210.0
1      420.0     297.0

It can also be passed to :class:`~skrub.SelectCols` or :class:`~skrub.DropCols`
to be embedded in scikit-learn pipelines:
