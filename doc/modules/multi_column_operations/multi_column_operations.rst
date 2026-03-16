.. currentmodule:: skrub

.. |ApplyToCols| replace:: :class:`ApplyToCols`
.. |ApplyToSubFrame| replace:: :class:`ApplyToSubFrame`
.. |SelectCols| replace:: :class:`SelectCols`
.. |DropCols| replace:: :class:`DropCols`
.. |s.string| replace:: :meth:`skrub.selectors.string`
.. |s.numeric| replace:: :meth:`skrub.selectors.numeric`
.. |RejectColumn| replace:: :class:`core.RejectColumn`
.. |ToDatetime| replace:: :class:`ToDatetime`
.. |SingleColumnTranformer| replace:: :class:`core.SingleColumnTranformer`
.. |StandardScaler| replace:: :class:`sklearn.preprocessing.StandardScaler`
.. |OneHotEncoder| replace:: :class:`sklearn.preprocessing.OneHotEncoder`

.. _user_guide_multiple_columns:

Operating over multiple columns at once
=======================================

Very often and for various reasons, transformers must be applied to multiple
columns at the same time. For example, all numeric columns in a dataframe may need
to be scaled at the same time.
While the heuristics used by the :class:`TableVectorizer` are usually good enough
to apply the proper transformers to different datatypes, using it may not be an
option in all cases. In scikit-learn pipelines, the column selection operation can
is done with the :class:`sklearn.compose.ColumnTransformer`.

Skrub provides alternative transformers that can achieve the same results:

- |ApplyToCols| maps a transformer to columns in a dataframe, so that all
  columns that satisfy a certain condition are transformed, while the others are
  left untouched.
- |SelectCols| allows specifying which columns should be kept.
- |DropCols| allows specifying the columns we want to discard.

|SelectCols| and |DropCols| can becombined with the skrub DataOps
to perform complex tasks
such as feature selection: refer to :ref:`user_guide_data_ops_feature_selection`
for more details.

All multi-column transformers provided by skrub can take skrub selectors as
parameters to have more control over the columns that are being transformed.
Skrub selectors are discussed at length in :ref:`user_guide_selectors`.

.. _apply_to_each_col:

Applying transformations to the columns with |ApplyToCols|
----------------------------------------------------------
|ApplyToCols| can be used to transform a subset of columns in a dataframe, while
leaving the non-selected columns unchanged. In this example, we want to apply
a |StandardScaler| to the numeric column, and a |OneHotEncoder| to the text column.

>>> import skrub.selectors as s
>>> from sklearn.pipeline import make_pipeline
>>> from skrub import ApplyToCols
>>> from sklearn.preprocessing import OneHotEncoder, StandardScaler
>>> import pandas as pd
>>> df = pd.DataFrame({"text": ["foo", "bar", "baz"], "number": [1, 2, 3]})


We use the |s.numeric| and |s.string| selectors to choose the respective columns:

>>> numeric = ApplyToCols(StandardScaler(), cols=s.numeric())
>>> string = ApplyToCols(OneHotEncoder(sparse_output=False), cols=s.string())

>>> transformed = make_pipeline(numeric, string).fit_transform(df)
>>> transformed
     number  text_bar  text_baz  text_foo
0 -1.224745       0.0       0.0       1.0
1  0.000000       1.0       0.0       0.0
2  1.224745       0.0       1.0       0.0

By default, |ApplyToCols| forwards all the selected columns together to the provided
transformer. For example, consider this dataframe:

>>> import numpy as np
>>> import pandas as pd
>>> df = pd.DataFrame(np.eye(4) * np.logspace(0, 3, 4), columns=list("abcd"))
>>> df
     a     b      c       d
0  1.0   0.0    0.0     0.0
1  0.0  10.0    0.0     0.0
2  0.0   0.0  100.0     0.0
3  0.0   0.0    0.0  1000.0

We want to apply a PCA with 2 components to it:

>>> from sklearn.decomposition import PCA
>>> pca = ApplyToCols(PCA(n_components=2))
>>> pca.fit_transform(df).round(2)
     pca0   pca1
0 -249.01 -33.18
1 -249.04 -33.68
2 -252.37  66.64
3  750.42   0.22

We can also apply the transformation only to a subset of the columns:

>>> pca = ApplyToCols(PCA(n_components=2), cols=["a", "b"])
>>> pca.fit_transform(df).round(2)
       c       d  pca0  pca1
0    0.0     0.0 -2.52  0.67
1    0.0     0.0  7.50  0.00
2  100.0     0.0 -2.49 -0.33
3    0.0  1000.0 -2.49 -0.33


If |ApplyToCols| is used with a transformer that inherits from
|SingleColumnTransformer|, or one that has the ``__single_column_transformer__``
attribute, then the transformer will be cloned and applied separately to each
column. Most skrub transformers belong to this category.

Here we want to apply |ToDatetime| to each of the datetime columns to convert
them to datetime dtype:

>>> df = pd.DataFrame({
...     'date_1': ['2024-01-15', '2024-02-20', '2024-03-10'],
...     'date_2': ['2023-12-01', '2024-01-05', '2024-02-28']
... })
>>> df
       date_1      date_2
0  2024-01-15  2023-12-01
1  2024-02-20  2024-01-05
2  2024-03-10  2024-02-28

|ApplyToCols| automatically detects that |ToDatetime| should be applied to each
column separately:

>>> from skrub._to_datetime import ToDatetime
>>> df_enc = ApplyToCols(ToDatetime()).fit_transform(df)
>>> df_enc
      date_1     date_2
0 2024-01-15 2023-12-01
1 2024-02-20 2024-01-05
2 2024-03-10 2024-02-28
>>> df_enc.dtypes
date_1    datetime64[...]
date_2    datetime64[...]
dtype: ...


Dealing with columns that cannot be handled by a transformer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

|ApplyToCols| can raise a |RejectColumn| exception if it cannot handle a specific
column:

>>> from skrub._to_datetime import ToDatetime
>>> df = pd.DataFrame(dict(birthday=["29/01/2024"], city=["London"]))
>>> df
    birthday    city
0  29/01/2024  London
>>> df.dtypes
birthday    ...
city        ...
dtype: ...
>>> ToDatetime().fit_transform(df["birthday"])
0   2024-01-29
Name: birthday, dtype: datetime64[...]
>>> ToDatetime().fit_transform(df["city"])
Traceback (most recent call last):
    ...
skrub._single_column_transformer.RejectColumn: Could not find a datetime format for column 'city'.

It is possible to change how rejected columns are handled through the ``allow_reject``
parameter.
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

Here, the column 'city' was rejected without being treated as an error: it was
was passed through unchanged and only ``birthday`` was converted to a
datetime column.

>>> transformed.dtypes
birthday    datetime64[...]
city                ...
dtype: ...

Renaming the transformed columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, |ApplyToCols| renames the transformed columns, and
removes the original features from the data. It is possible to rename the columns
by providing a formatting string to the ``rename_columns`` parameter:

>>> from sklearn.preprocessing import StandardScaler
>>> df = pd.DataFrame(dict(A=[-10., 10.], B=[0., 100.]))
>>> scaler = ApplyToCols(StandardScaler(), rename_columns='{}_scaled')
>>> scaler.fit_transform(df)
    A_scaled  B_scaled
0      -1.0      -1.0
1       1.0       1.0

By setting ``keep_original=True``, the starting columns are not dropped from the
transformed dataframe. The ``rename_columns`` parameter can be used to avoid
name collisions:

>>> scaler = ApplyToCols(
...     StandardScaler(), keep_original=True, rename_columns="{}_scaled"
... )
>>> scaler.fit_transform(df)
      A      B  A_scaled  B_scaled
0 -10.0    0.0      -1.0      -1.0
1  10.0  100.0       1.0       1.0
