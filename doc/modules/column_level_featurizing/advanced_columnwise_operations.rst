.. currentmodule:: skrub

.. |ApplyToCols| replace:: :class:`ApplyToCols`
.. |RejectColumn| replace:: :class:`core.RejectColumn`
.. |SingleColumnTranformer| replace:: :class:`core.SingleColumnTranformer`

.. _user_guide_single_column_transformer:

Advanced columnwise operations
------------------------------

The single column transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In cases where we want to apply a custom transformation to a series we need the |ApplyToCols|
structure to handle multiple columns, and if this transformation needs to be able to reject certain
columns and communicate this to |ApplyToCols|, we must to create a transformer from scratch
that raises this exception when appropriate.

Hence the ``SingleColumnTransformer`` class. It is originally a base class from which many transformers are
inherited, but it can also be used to create new transformers. As long we specify
corresponding ``fit`` and ``transform`` methods to our custom transformer, it can be passed on like any other.

For instance, we might want to create a custom transformer specialized in parsing zip codes of a certain
format, that raises |RejectColumn| when the length of the provided
zip code is incorrect:

>>> from skrub.core import RejectColumn, SingleColumnTransformer
>>> import pandas as pd
>>>
>>> class ZipcodeParser(SingleColumnTransformer):
...     def __init__(self):
...         return
...     def fit_transform(self, X, y=None):
...         if any(X.map(len) != 5):
...             raise RejectColumn('This transformer only takes zip codes of length 5.')
...         else:
...             letters = X.map(lambda s: s[:2])
...             try:
...                 numbers = X.map(lambda s: int(s[2:]))
...             except:
...                 raise RejectColumn('Input zip codes must consist of two letters followed by three numbers.')
...             return(pd.DataFrame({'letters': letters, 'numbers': numbers}))
>>> df = pd.DataFrame({'sent': ["AB123", "BD601", "HS014"], 'received': ["AB1C45", "DU3K93", "WB9M88"]})
>>> df
    sent received
0  AB123   AB1C45
1  BD601   DU3K93
2  HS014   WB9M88
>>> ZipcodeParser().fit_transform(df["sent"])
  letters  numbers
0      AB      123
1      BD      601
2      HS       14
>>> ZipcodeParser().fit_transform(df["received"])
Traceback (most recent call last):
    ...
skrub.core.RejectColumn: This transformer only takes zip codes of length 5.




Rejection handling with ``ApplyToCols``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|ApplyToCols| allows flexible manipulation of dataframes by automatically
applying any of Skrub's columnwise transformers to multiple columns in any given dataframe.
More information can be found on this class's basic usage on :ref:`user_guide_multiple_columns`.

If the input columns are unable to be transformed, a specific exception exists to indicate
this: |RejectColumn|.

>>> from skrub import ToDatetime
>>> df = pd.DataFrame(dict(birthday=["29/01/2024"], city=["London"]))
>>> df
        birthday    city
0  29/01/2024  London
>>> df.dtypes
birthday    ...
city        ...
dtype: object
>>> ToDatetime().fit_transform(df["birthday"])
0   2024-01-29
Name: birthday, dtype: datetime64[...]
>>> ToDatetime().fit_transform(df["city"])
Traceback (most recent call last):
    ...
skrub.core.RejectColumn: Could not find a datetime format for column 'city'.

The ``allow_reject`` parameter in |ApplyToCols| specifies how to react if such an exception is raised.
By default, no special handling is performed and rejections are considered to be errors:

>>> from skrub import ApplyToCols
>>> to_datetime = ApplyToCols(ToDatetime())
>>> to_datetime.fit_transform(df)
Traceback (most recent call last):
    ...
ValueError: Transformer ToDatetime.fit_transform failed on column 'city'. See above for the full traceback.

However, setting ``allow_reject=True`` gives the transformer itself some
control over which columns it should be applied to. For example, whether a
string column contains dates is only known once we try to parse them.
Therefore it might be sensible to try to parse all string columns but allow
the transformer to reject those that, upon inspection, do not contain dates.

>>> to_datetime = ApplyToCols(ToDatetime(), allow_reject=True)
>>> transformed = to_datetime.fit_transform(df)
>>> transformed
    birthday    city
0 2024-01-29  London

Now the column 'city' was rejected but this was not treated as an error;
'city' was passed through unchanged and only 'birthday' was converted to a
datetime column.

>>> transformed.dtypes
birthday    datetime64[...]
city                ...
dtype: object
>>> to_datetime.transformers_
{'birthday': ToDatetime()}
