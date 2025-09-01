Type of selectors
-----------------

:func:`~skrub.selectors.all` is another simple selector, especially useful for default
arguments since it keeps all columns:

>>> SelectCols(cols=s.all()).fit_transform(df)
   height_mm  width_mm kind  ID
0      297.0     210.0   A4   4
1      420.0     297.0   A3   3

Selectors can be combined with operators, for example if we wanted all columns
except the "mm" columns above:

>>> SelectCols(s.all() - s.cols("height_mm", "width_mm")).fit_transform(df)
  kind  ID
0   A4   4
1   A3   3

This module provides several kinds of selectors, which allow to select columns by
name, data type, contents, or according to arbitrary user-provided rules.

>>> SelectCols(s.numeric()).fit_transform(df)
   height_mm  width_mm  ID
0      297.0     210.0   4
1      420.0     297.0   3

>>> SelectCols(s.glob('*_mm')).fit_transform(df)
   height_mm  width_mm
0      297.0     210.0
1      420.0     297.0
