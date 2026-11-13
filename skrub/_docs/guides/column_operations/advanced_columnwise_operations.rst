.. currentmodule:: skrub

.. |ApplyToCols| replace:: :class:`ApplyToCols`
.. |RejectColumn| replace:: :class:`~core.RejectColumn`
.. |SingleColumnTransformer| replace:: :class:`~core.SingleColumnTransformer`
.. |ToDatetime| replace:: :class:`ToDatetime`

.. _user_guide_single_column_transformer:

How to define your own column-wise transformer
----------------------------------------------

.. _single_column_transformer:

The single column transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you have a transformation that only needs to see one column at a time, you
can define a transformer that processes a single column, then let |ApplyToCols|
take care of mapping it to all the relevant columns in the input dataframe.
This can be done by defining the transformer as a subclass of
|SingleColumnTransformer|. Then, the transformer can restrict the set of columns
to which it is applied.

For example, consider a situation where we want to identify municipalities in France
based on their "COG"
(`Code officiel géographique <https://en.wikipedia.org/wiki/INSEE_code#Geographical_codes>`_).
The COG is a 5-digit number with the format XXYYY,
where the XX digits report the number of the department, and the YYY contain the
code of the municipality.
Columns that do not satisfy this format should be ignored.

>>> import pandas as pd
>>> df = pd.DataFrame(
...     {
...         "sent": ["75001", "13001", "69002"],
...         "received": ["91120", "12110", "88150"],
...         "notes": ["Sent", "Missing", "Sent"],
...     }
... )
>>> df
    sent received    notes
0  75001    91120     Sent
1  13001    12110  Missing
2  69002    88150     Sent

We would like to be able to "unpack" the code so that we have a column for the
department code and one for the municipality code. The transformer should also be able to handle columns
that do not satisfy the format we specify by "rejecting" them.
A "rejected" column should pass through unchanged, as it cannot be handled
by this particular transformer.

|SingleColumnTransformer| and |RejectColumn| let us define a transformer that satisfies these
requirements:

>>> from skrub.core import RejectColumn, SingleColumnTransformer
>>> class GeocodeParser(SingleColumnTransformer):
...     def fit_transform(self, X, y=None):
...         return self.transform(X)
...
...     def transform(self, X, y=None):
...         self.col_name_ = X.name
...         extracted = X.str.extractall(r"^(?P<dep>\d{2})(?P<city>\d{3})$").reset_index(
...             drop=True
...         )
...         if extracted.shape[0] != X.shape[0]:
...             raise RejectColumn("Incorrect format")
...         return pd.DataFrame(
...             {
...                 f"{self.col_name_}_department": extracted["dep"],
...                 f"{self.col_name_}_commune": extracted["city"],
...             }
...         )

>>> GeocodeParser().fit_transform(df["sent"])
  sent_department sent_commune
0              75          001
1              13          001
2              69          002

We can use |ApplyToCols| to apply this transformer to the entire dataframe at once,
and set ``allow_reject=True`` to let rejected columns pass through without changes:

>>> from skrub import ApplyToCols
>>> ApplyToCols(GeocodeParser(), allow_reject=True).fit_transform(df)
sent_department sent_commune received_department received_commune    notes
0              75          001                  91              120     Sent
1              13          001                  12              110  Missing
2              69          002                  88              150     Sent

Note how the ``"received"`` column has been "rejected" and passed through unmodified.

The |SingleColumnTransformer| is designed to work in conjunction with |ApplyToCols|
and the skrub :ref:`selectors <user_guide_selectors>` to provide a high degree of
control over what columns should be modified.


How to reject or ignore columns with |ApplyToCols| and |RejectColumn|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The combination of |ApplyToCols| and |RejectColumn| allows flexible manipulation
and error checking of dataframes.

When passed a column that it cannot handle, a |SingleColumnTransformer| should raise
a |RejectColumn| exception to signal it should not be applied to that column. It
is then up to |ApplyToCols| to decide whether the column should be passed through
unchanged or to propagate the exception, based on the ``allow_reject`` parameter.

>>> ApplyToCols(GeocodeParser()).fit_transform(df)  # doctest: +SKIP
Traceback (most recent call last):
    ...
RejectColumn: This transformer only takes zip codes of length 5.
Transformer GeocodeParser.fit_transform failed on column 'notes'. See above for the full traceback.

In some cases, we may want to let columns through because we don't know which
columns will fail. For example, we may want to use |ToDatetime| to find datetime
columns: in this case, we don't know in advance which columns should be chosen.

>>> from skrub import ToDatetime
>>> df = pd.DataFrame(dict(birthday=["29/01/2024"], city=["London"]))
>>> df
        birthday    city
0  29/01/2024  London
>>> df.dtypes
birthday    ...
city        ...
dtype: object

Converting the datetime column works:

>>> ToDatetime().fit_transform(df["birthday"])
0   2024-01-29
Name: birthday, dtype: datetime64[...]

While non-datetimes would raise |RejectColumn|:

>>> ToDatetime().fit_transform(df["city"])
Traceback (most recent call last):
    ...
skrub.core.RejectColumn: Could not find a datetime format for column 'city'.

The ``allow_reject`` parameter in |ApplyToCols| allows the same transformer
to be applied to all columns without having to worry about which columns will actually be converted,
any rejected column is passed through unchanged.
Here, |ToDatetime| is applied only to the "birthday" column, while "city" is passed
through unchanged and no exception is raised.

>>> to_datetime = ApplyToCols(ToDatetime(), allow_reject=True)
>>> transformed = to_datetime.fit_transform(df)
>>> transformed
    birthday    city
0 2024-01-29  London

We can see that the only column that has a transformer is "birthday":

>>> to_datetime.transformers_
{'birthday': ToDatetime()}
