.. _user_guide_selectors:

Skrub Selectors, for selecting columns in a dataframe
=====================================================

In skrub, a selector represents a column selection rule, such as "all columns
that have numeric data types, except the column ``'User ID'``".

Selectors have two main benefits:

- Expressing complex selection rules in a simple and concise way by combining
  selectors with operators. A range of useful selectors is provided by this module.
- Delayed selection: passing a selection rule which will evaluated later on a dataframe
  that is not yet available. For example, without selectors, it is not possible to
  instantiate a :class:`~skrub.SelectCols` that selects all columns except those with
  the suffix 'ID' if the data on which it will be fitted is not yet available.

Introduction to selectors
------------------------------

Here is an example dataframe. Note that selectors support both Pandas and Polars
dataframes::

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
column names::

    >>> from skrub import selectors as s
    >>> mm_cols = s.cols('height_mm', 'width_mm')
    >>> mm_cols
    cols('height_mm', 'width_mm')

Using selectors:

* **select function**: the above selector can be passed to the :func:`~skrub.selectors.select` function::

    >>> s.select(df, mm_cols)
    height_mm  width_mm
    0      297.0     210.0
    1      420.0     297.0

* **transformers**: various transformers in skrub use selectors to select and transform columns
  in a scikit-learn pipeline: :class:`~skrub.ApplyToCols`, :class:`~skrub.ApplyToFrame`,
  :class:`~skrub.DropCols`, :class:`~skrub.SelectCols`, as
  :ref:`detailed below <selectors_and_transformer>`.

* **DataOps** selectors can be passed to
  :ref:`skrub DataOps <user_guide_data_ops_index>` when applying an
  estimator with the :func:`skrub.DataOp.skb.apply` function::

    >>> import skrub
    >>> from sklearn.preprocessing import StandardScaler
    >>> skrub.X(df).skb.apply(StandardScaler(), cols=mm_cols)
    <Apply StandardScaler>
    Result:
    ―――――――
    kind  ID  height_mm  width_mm
    0   A4   4       -1.0      -1.0
    1   A3   3        1.0       1.0

Type of selectors
-----------------

:func:`~skrub.selectors.all` is another simple selector, especially useful for default
arguments since it keeps all columns::

    >>> from skrub import SelectCols
    >>> SelectCols(cols=s.all()).fit_transform(df)
    height_mm  width_mm kind  ID
    0      297.0     210.0   A4   4
    1      420.0     297.0   A3   3

Selectors can be combined with operators, for example if we wanted all columns
except the "mm" columns above::

    >>> SelectCols(s.all() - s.cols("height_mm", "width_mm")).fit_transform(df)
    kind  ID
    0   A4   4
    1   A3   3

This module provides several kinds of selectors, which allow to select columns by
name, data type, contents, or according to arbitrary user-provided rules::

    >>> SelectCols(s.numeric()).fit_transform(df)
    height_mm  width_mm  ID
    0      297.0     210.0   4
    1      420.0     297.0   3

    >>> SelectCols(s.glob('*_mm')).fit_transform(df)
    height_mm  width_mm
    0      297.0     210.0
    1      420.0     297.0

.. seealso::

    * :ref:`selectors_details` explains more the various selectors

    * :ref:`selectors_ref` gives the exhaustive list of selectors

    * :ref:`user_guide_advanced_selectors`

Combining selectors
-------------------

The available operators are ``|``, ``&``, ``-``, ``^`` with the meaning of usual
python sets, and ``~`` to invert a selection::

    >>> SelectCols(s.glob('*_mm')).fit_transform(df)
    height_mm  width_mm
    0      297.0     210.0
    1      420.0     297.0

    >>> SelectCols(~s.glob('*_mm')).fit_transform(df)
    kind  ID
    0   A4   4
    1   A3   3

    >>> SelectCols(s.glob('*_mm') | s.cols('ID')).fit_transform(df)
    height_mm  width_mm  ID
    0      297.0     210.0   4
    1      420.0     297.0   3

    >>> SelectCols(s.glob('*_mm') & s.glob('height_*')).fit_transform(df)
    height_mm
    0      297.0
    1      420.0

    >>> SelectCols(s.glob('*_mm') ^ s.string()).fit_transform(df)
    height_mm  width_mm kind
    0      297.0     210.0   A4
    1      420.0     297.0   A3

The operators respect the usual short-circuit rules. For example, the
following selector won't compute the cardinality of non-categorical columns::

    >>> s.categorical() & s.cardinality_below(10)
    (categorical() & cardinality_below(10))

.. _selectors_and_transformer:

Using selectors with other skrub transformers
-------------------------------------------------

Skrub transformers are designed to be used in conjunction with other transformers
that operate on columns to improve their versatility.

For example, we can drop columns that have more unique values than a certain amount
by combining :func:`~skrub.selectors.cardinality_below` with :class:`skrub.DropCols`.
We first select the columns that have more than 3 unique values, then we invert the
selector and finally transform the dataframe.

>>> df = pd.DataFrame({
... "not a lot": [1, 1, 1, 2, 2],
... "too_many":  [1,2,3,4,5]})

>>> from skrub import DropCols
>>> DropCols(cols=~s.cardinality_below(3)).fit_transform(df)
   not a lot
0          1
1          1
2          1
3          2
4          2

Selectors can be used in conjunction with :class:`~skrub.ApplyToCols` to transform columns
based on specific requirements.

Consider the following example:

>>> import pandas as pd
>>> data = {
...     "subject": ["Math", "English", "History", "Science", "Art"],
...     "grade": [5, 4, 3, 4, 3]
... }
>>> df = pd.DataFrame(data)
>>> df
   subject grade
0     Math     5
1  English     4
2  History     3
3  Science     4
4      Art     3

We might want to apply the :class:`~sklearn.preprocessing.StandardScaler` only to the numeric column. We can
do this like this:

>>> from skrub import ApplyToCols
>>> from sklearn.preprocessing import StandardScaler
>>> ApplyToCols(StandardScaler(), cols=s.numeric()).fit_transform(df)
   subject     grade
0     Math  1.603567
1  English  0.267261
2  History -1.069045
3  Science  0.267261
4      Art -1.069045
