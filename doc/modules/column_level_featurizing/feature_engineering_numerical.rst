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
which standardizes all numeric-like columns to ``float32`` and handles a wide
range of real-world formatting issues automatically.

The |ToFloat| transformer is used internally by both the |Cleaner| and the
|TableVectorizer| to guarantee that downstream estimators receive clean and
uniform numeric data.

What |ToFloat| does
-------------------

The |ToFloat| transformer provides:

- **Automatic conversion to 32-bit floating-point values (`float32`).**
  This dtype is lightweight and fully supported by scikit-learn estimators.

- **Automatic parsing of decimal and thousands separators**, regardless of locale:
  - The decimal separator must be specified explicitly and can be either ``.`` or ``,``
  - The thousands separator can be one of ``.``, ``,``, space (``" "``), apostrophe (``'``),
  or None (no thousands separator)
  - The transformer supports integers, decimals (including leading-decimal forms such as .56 or ,56), scientific notation
  and negative numbers
  - Numbers in parentheses are interpreted as negative numbers (``(1,234.56)`` → ``-1234.56``). This format is more common in financial datasets.
  - Decimal and thousands separators must be different characters

- **Scientific notation parsing** (e.g. ``1.23e+4``)

- **Graceful handling of invalid or non-numeric values during transform**:
  - During ``fit``: non-convertible values raise a ``RejectColumn`` exception
  - During ``transform``: invalid entries become ``NaN`` instead of failing

- **Rejection of categorical and datetime columns**, which should not be cast to numeric.

As with all skrub transformers, |ToFloat| behaves like a standard
scikit-learn transformer and is fully compatible with pipelines.

How to use |ToFloat|
--------------------
The |ToFloat| transformer must be applied to individual columns, and it behaves
like a standard scikit-learn transformer.
|ToFloat| requires a ``decimal`` and a ``thousands`` separator, which are ``'.'`` and
``None`` (no thousands separator) by default.
Each column is expected to use a single separator for decimals, and one for thousands:
if any characters other than the provided selectors are encountered in the column, it will not
be converted.

During ``fit``, |ToFloat| attempts to convert all values in the column to
numeric values after automatically removing other possible thousands separators
(``,``, ``.``, space, apostrophe). If any value cannot be converted, the column
is rejected with a ``RejectColumn`` exception.

During ``transform``, invalid or non-convertible values are replaced by ``NaN``
instead of raising an error.

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

Locale-dependent decimal separators can be handled by specifying the
``decimal`` and ``thousand`` parameter. Here we use comma as decimal separator, and
a space as thousands separators:

>>> s = pd.Series(["4 567,89", "12 567,89"], name="x")
>>> ToFloat(decimal=",", thousand=" ").fit_transform(s)
0    4567.8...
1    12567.8...
Name: x, dtype: float32

Parentheses interpreted as negative numbers:

>>> s = pd.Series(["-1,234.56", "(1,234.56)"], name="neg")
>>> ToFloat(thousand=",").fit_transform(s)
0   -1234.5...
1   -1234.5...
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
skrub._single_column_transformer.RejectColumn: Could not convert column 'x' to numbers.


During ``transform``, invalid entries become ``NaN`` instead of raising an error:
>>> s = pd.Series(['1.1', '2.2'], name='x')
>>> to_float = ToFloat(decimal=".")
>>> to_float.fit_transform(s)
0    1.1
1    2.2
Name: x, dtype: float32

>>> to_float.transform(pd.Series(['3.3', 'invalid'], name='x'))
0    3.3
1    NaN
Name: x, dtype: float32
