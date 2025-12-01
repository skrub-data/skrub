.. |ToFloat| replace:: :class:`~skrub.ToFloat`
.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |Cleaner| replace:: :class:`~skrub.Cleaner`

.. _user_guide_feature_engineering_numeric_to_float:

Converting heterogeneous numeric values to uniform float32
==========================================================

Many tabular datasets contain numeric information stored as strings, mixed
representations, locale-specific formats, or other non-standard encodings.
Common issues include:

- Thousands separators (``1,234.56`` or ``1 234,56``)
- Use of apostrophes as separators (``4'567.89``)
- Negative numbers encoded inside parentheses (``(1,234.56)``)
- String columns that contain mostly numeric values, but with occasional invalid entries

To provide consistent numeric behavior, skrub includes the |ToFloat| transformer,
which **standardizes all numeric-like columns to ``float32``** and handles a wide
range of real-world formatting issues automatically.

The |ToFloat| transformer is used internally by both the |Cleaner| class and the
|TableVectorizer| to guarantee that downstream estimators receive clean and
uniform numeric data.

What |ToFloat| does
-------------------

The |ToFloat| transformer provides:

- **Automatic conversion to 32-bit floating-point values (`float32`).**
  This dtype is lightweight and fully supported by scikit-learn estimators.

- **Automatic parsing of decimal separators**, regardless of locale:
  - ``.`` or ``,`` can be used as decimal point
  - thousands separators (``.``, ``,``, space, apostrophe) are removed automatically

- **Parentheses interpreted as negative numbers**, a common format in financial datasets:
  - ``(1,234.56)`` → ``-1234.56``

- **Scientific notation parsing** (e.g. ``1.23e+4``)

- **Graceful handling of invalid or non-numeric values during transform**:
  - During ``fit``: non-convertible values raise a ``RejectColumn`` exception
  - During ``transform``: invalid entries become ``NaN`` instead of failing

- **Rejection of categorical and datetime columns**, which should not be cast to numeric.

As with all skrub transformers, |ToFloat| behaves like a standard
scikit-learn transformer and is fully compatible with pipelines.

Examples
--------

Parsing numeric-formatted strings:

>>> import pandas as pd
>>> from skrub import ToFloat
>>> s = pd.Series(['1.1', None, '3.3'], name='x')
>>> ToFloat().fit_transform(s)
0    1.1
1    NaN
2    3.3
Name: x, dtype: float32

Automatic handling of locale-dependent decimal separators:

>>> s = pd.Series(["4 567,89", "4'567,89"], name="x")
>>> ToFloat(decimal=",").fit_transform(s)   # doctest: +SKIP
0    4567.89
1    4567.89
Name: x, dtype: float32

Parentheses interpreted as negative numbers:

>>> s = pd.Series(["-1,234.56", "(1,234.56)"], name="neg")
>>> ToFloat().fit_transform(s)   # doctest: +SKIP
0   -1234.56
1   -1234.56
Name: neg, dtype: float32

Scientific notation:

>>> s = pd.Series(["1.23e+4", "1.23E+4"])
>>> ToFloat(decimal=".").fit_transform(s)
0    12300.0
1    12300.0
dtype: float32

Columns that cannot be converted are rejected during ``fit``:

>>> s = pd.Series(['1.1', 'hello'], name='x')
>>> ToFloat(decimal=".").fit_transform(s)
Traceback (most recent call last):
    ...
skrub._apply_to_cols.RejectColumn: Could not convert column 'x' to numbers.

How |ToFloat| is used in skrub
------------------------------

The |ToFloat| transformer is used internally in:

- the **Cleaner** (|Cleaner|), to normalize all numeric-like columns before modeling
- the **|TableVectorizer|**, ensuring a consistent numeric dtype across all numeric features

This makes |ToFloat| a core building block of skrub’s handling of heterogeneous
tabular data.

``ToFloat`` ensures that downstream machine-learning models receive numeric data
that is clean, consistent, lightweight, and free of locale-specific quirks or
string-encoded values.
