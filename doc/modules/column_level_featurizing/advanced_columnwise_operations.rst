Advanced columnwise operations
------------------------------


Rejected columns
----------------

The ``RejectColumn`` exception exists to indicate when a transformer cannot handle a
given column.

>>> from skrub import ToDatetime
>>> import pandas as pd
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
skrub.core._single_column_transformer.RejectColumn: Could not find a datetime format for column 'city'.

How these rejections are handled depends on the ``allow_reject`` parameter.
By default, no special handling is performed and rejections are considered
to be errors:

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


The single column transformer
-----------------------------

In cases where the user needs finer control over a custom transformer's behavior on different columns,
or if a workflow involves a non-skrub transformer which doesn't handle column rejection, it is necessary
to create a transformer from scratch that is capable of handling this exception.

Hence the ``SingleColumnTransformer`` class. It is originally a base class from which many transformers are
inherited, but it can also be used to create new transformers. Custom transformers inherited from this class
can serve as wrappers for non-Skrub transformers. For instance, if one wanted to implement the ``RejectColumn``
exception into scikit-learn's ``StandardScaler``, one could create a ``NewScaler`` as follows::

>>> from skrub.core import RejectColumn, SingleColumnTransformer
>>> from sklearn.preprocessing import StandardScaler
>>> class Scaler(SingleColumnTransformer):
...     def __init__(self):
...         self.scaler = StandardScaler()
...
...     def fit_transform(self, x, y=None):
...         if x.dtype != 'float64':
...             raise RejectColumn('This scaler only takes float64 types.')
...         else:
...             return(pd.DataFrame(self.scaler.fit_transform(pd.DataFrame(x), y)))
...
...     def transform(self, x):
...         return(pd.DataFrame(self.scaler.transform(pd.DataFrame(x))))
>>> df = pd.DataFrame(dict(product=["Chair", "Table", "Bed"], price=[30.0, 60.0, 200.0]))
>>> Scaler().fit_transform(df["price"])
          0
0 -0.899843
1 -0.494913
2  1.394756
>>> Scaler().fit_transform(df["product"])
Traceback (most recent call last):
    ...
skrub.core._single_column_transformer.RejectColumn: This scaler only takes float64 types.




A custom class inherited from a ``SingleColumnTransformer`` can also enable to run a custom criterion to
check whether or not to accept a column::

>>> class NewScaler(SingleColumnTransformer):
...     def __init__(self):
...         self.scaler = StandardScaler()
...
...     def fit_transform(self, x, y=None):
...         if not check(x):
...             raise RejectColumn
...         else:
...             self.scaler.fit_transform(x, y)
...
...     def transform(self, x):
...         self.scaler.fit_transform(x)

Where ``check`` is a specially-defined function.
