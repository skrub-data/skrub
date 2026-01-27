Advanced columnwise operations
------------------------------


Independent columnwise operations
---------------------------------

When ``ApplyToCols``` is called, what is being run under the hood is in fact a series of
transformers known as ``SingleColumnTransformer``. If the transformer only needs to be run on
a single column, it can be given the ``__single_column_transformer__`` attribute. However,
there are cases where it is useful to work directly with the ``SingleColumnTransformer``
itself, as it avoids some boilerplate:

    - The required ``__single_column_transformer__`` attribute is set.
    - ``fit`` is defined (calls ``fit_transform`` and discards the result).
    - ``fit``, ``transform`` and ``fit_transform`` are wrapped to check
        that the input is a single column and raise a ``ValueError`` with a
        helpful message when it is not.
    - A note about single-column transformers (vs dataframe transformers)
        is added after the summary line of the docstring.

***TO BE COMPLETED***



Rejected columns
----------------

The transformer can raise ``RejectColumn`` to indicate it cannot handle a
given column.

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
skrub.core._single_column_transformer.RejectColumn: Could not find a datetime format for column 'city'.

How these rejections are handled depends on the ``allow_reject`` parameter.
By default, no special handling is performed and rejections are considered
to be errors:

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
