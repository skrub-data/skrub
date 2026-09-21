.. currentmodule:: skrub

.. |ApplyToCols| replace:: :class:`ApplyToCols`
.. |RejectColumn| replace:: :class:`core.RejectColumn`
.. |SingleColumnTranformer| replace:: :class:`core.SingleColumnTranformer`
.. |ToDatetime| replace:: :class:`ToDatetime`

.. _user_guide_single_column_transformer:

Advanced columnwise operations
------------------------------

.. _single_column_transformer:

The single column transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are situations in which information in a column may be encoded according
to a specific system, and it may be beneficial to write a transformer that automatically
converts columns that satisfy the format into separate columns for further processing.

The |SingleColumnTranformer| can be used to define such a transformer, providing
additional features to simplify its inclusion in a pipeline and the rejection of
columns that cannot be handled by the transformer.

We can use the code used to identify municipalities in France
(`COG, Code officiel géographique <https://en.wikipedia.org/wiki/INSEE_code#Geographical_codes>`_)
as an example of this problem. The COG is a 5-digit number with the format XXYYY,
where the XX digits report the number of the department, and the YYY contain the
code of the municipality.

>>> import pandas as pd
>>> df = pd.DataFrame({'sent': ["75001", "13001", "69002"], 'received': ["ABCDE", "DU3K93", "WB9M88"]})
>>> df
    sent received
0  75001    ABCDE
1  13001   DU3K93
2  69002   WB9M88

We would like to be able to "unpack" the code so that we have a column for the
department code and one for the commune code; the transformer should also be able to handle columns
that do not satisfy the format we specify by "rejecting" them.
A "rejected" column should be passed through unchanged, as it cannot be handled
by this particular transformer.

|SingleColumnTranformer| and |RejectColumn| let us define a transformer that satisfies these
requirements:

>>> from skrub.core import RejectColumn, SingleColumnTransformer
>>> class ZipcodeParser(SingleColumnTransformer):
...     def __init__(self):
...         return
...     def fit_transform(self, X, y=None):
...         self.col_name_ = X.name if X.name else "parsed_zip"
...         if any(X.map(len) != 5):
...             raise RejectColumn('This transformer only takes zip codes of length 5.')
...         if not all(X.map(lambda s: s.isdigit())):
...             raise RejectColumn('Input zip codes must be numeric.')
...         department = X.map(lambda s: s[:2])
...         commune = X.map(lambda s: s[2:])
...         return(pd.DataFrame({f'{self.col_name_}_department': department,
...                              f'{self.col_name_}_commune': commune}))
...     def transform(self, X, y=None):
...         department = X.map(lambda s: s[:2])
...         commune = X.map(lambda s: s[2:])
...         return(pd.DataFrame({f'{self.col_name_}_department': department,
...                              f'{self.col_name_}_commune': commune}))


>>> ZipcodeParser().fit_transform(df["sent"])
  sent_department sent_commune
0              75          001
1              13          001
2              69          002

We can use |ApplyToCols| to apply this transformer to the entire dataframe at once,
and set ``allow_reject=True`` to let rejected columns through without changes:

>>> from skrub import ApplyToCols
>>> ApplyToCols(ZipcodeParser(), allow_reject=True).fit_transform(df)
  sent_department sent_commune received
0              75          001    ABCDE
1              13          001   DU3K93
2              69          002   WB9M88

Note how the ``"received"`` column has been "rejected" and passed through unmodified.



Rejection handling with |ApplyToCols| and |RejectColumn|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The combination |ApplyToCols| and |RejectColumn| allows allows flexible manipulation
and error checking of dataframe. In the previous example, we decided to ignore the
malformed ``"received"`` column by setting ``allow_reject=True``. If, however,
we want our transformer to fail if it encounters a column that it cannot parse,
we can keep the default value of ``allow_reject=False``, so that the transform
fails as soon as a malformed column is encountered:

>>> ApplyToCols(ZipcodeParser()).fit_transform(df)  # doctest: +SKIP
Traceback (most recent call last):
    ...
skrub.core.RejectColumn: Input zip codes must be numeric.
Transformer ZipcodeParser.fit_transform failed on column 'received'. See above for the full traceback.
Letting rejected columns through can be useful for situations in which we do not
know the content of a column in advance, like when we are trying to convert to
datetime columns in a dataframe, without knowing which ones actually contain dates.

>>> from skrub import ToDatetime
>>> df = pd.DataFrame(dict(birthday=["29/01/2024"], city=["London"]))
>>> df
        birthday    city
0  29/01/2024  London
>>> df.dtypes
birthday    ...
city        ...
dtype: object

Converting a datetime column would work:

>>> ToDatetime().fit_transform(df["birthday"])
0   2024-01-29
Name: birthday, dtype: datetime64[...]

While non-datetimes would raise |RejectColumn|:

>>> ToDatetime().fit_transform(df["city"])
Traceback (most recent call last):
    ...
skrub.core.RejectColumn: Could not find a datetime format for column 'city'.

The ``allow_reject`` parameter in |ApplyToCols| allows to apply the same transformer
to all columns without having to worry about which columns will actually be converted:
here, |ToDatetime| is applied only to the "birthday" column, while "city" is passed
through unchanged and no exception is raised.

>>> to_datetime = ApplyToCols(ToDatetime(), allow_reject=True)
>>> transformed = to_datetime.fit_transform(df)
>>> transformed
    birthday    city
0 2024-01-29  London

We can see that the only column that has a transformer is "birthday":

>>> to_datetime.transformers_
{'birthday': ToDatetime()}
