.. currentmodule:: skrub

.. |ApplyToCols| replace:: :class:`ApplyToCols`
.. |ApplyToFrame| replace:: :class:`ApplyToFrame`
.. |SelectCols| replace:: :class:`SelectCols`
.. |DropCols| replace:: :class:`DropCols`
.. _user_guide_multiple_columns:

Operating over multiple columns at once
=======================================

Very often and for various reasons, transformers must be applied to multiple
columns at the same time. For example, all numeric columns in a dataframe may need
to be scaled at the same time.
While the heuristics used by the :class:`TableVectorizer` are usually good enough
to apply the proper transformers to different datatypes, using it may not be an
option in all cases. In scikit-learn pipelines, the column selection operation can
is done with the :class:`sklearn.compose.ColumnTransformer`:


>>> import pandas as pd
>>> from sklearn.compose import make_column_selector as selector
>>> from sklearn.compose import make_column_transformer
>>> from sklearn.preprocessing import StandardScaler, OneHotEncoder
>>>
>>> df = pd.DataFrame({"text": ["foo", "bar", "baz"], "number": [1, 2, 3]})
>>>
>>> categorical_columns = selector(dtype_include=object)(df)
>>> numerical_columns = selector(dtype_exclude=object)(df)
>>>
>>> ct = make_column_transformer(
...       (StandardScaler(),
...        numerical_columns),
...       (OneHotEncoder(handle_unknown="ignore"),
...        categorical_columns))
>>> transformed = ct.fit_transform(df)
>>> transformed
array([[-1.22474487,  0.        ,  0.        ,  1.        ],
       [ 0.        ,  1.        ,  0.        ,  0.        ],
       [ 1.22474487,  0.        ,  1.        ,  0.        ]])

Skrub provides alternative transformers that can achieve the same results:

- |ApplyToCols| maps a transformer to columns in a dataframe, so that all
  columns that satisfy a certain condition are transformed, while the others are
  left untouched.
- |ApplyToFrame| applies a transformer to a collection of columns *at once*.
  This is different from |ApplyToCols|, which instead transforms each column
  one at a time.
- |SelectCols| allows specifying which columns should be kept.
- |DropCols| allows specifying the columns we want to discard.

|SelectCols| and |DropCols| can becombined with the skrub DataOps
to perform complex tasks
such as feature selection: refer to :ref:`user_guide_data_ops_feature_selection`
for more details.

All multi-column transformers provided by skrub can take skrub selectors as
parameters to have more control over the columns that are being transformed.
Skrub selectors are discussed at length in :ref:`user_guide_selectors`.


Applying transformations to the columns with |ApplyToCols| and |ApplyToFrame|
-----------------------------------------------------------------------------
|ApplyToCols| can be used to transform a subset of columns in a dataframe, while
leaving the remaining columns unchanged. It simplifies operations such as the
example above, which can be rewritten with |ApplyToCols| as follows:

>>> import skrub.selectors as s
>>> from sklearn.pipeline import make_pipeline
>>> from skrub import ApplyToCols
>>>
>>> numeric = ApplyToCols(StandardScaler(), cols=s.numeric())
>>> string = ApplyToCols(OneHotEncoder(sparse_output=False), cols=s.string())
>>>
>>> transformed = make_pipeline(numeric, string).fit_transform(df)
>>> transformed
   text_bar  text_baz  text_foo    number
0       0.0       0.0       1.0 -1.224745
1       1.0       0.0       0.0  0.000000
2       0.0       1.0       0.0  1.224745

|ApplyToCols| can raise a ``RejectColumn`` exception if it cannot handle a specific
column:

>>> from skrub._to_datetime import ToDatetime
>>> df = pd.DataFrame(dict(birthday=["29/01/2024"], city=["London"]))
>>> df
    birthday    city
0  29/01/2024  London
>>> df.dtypes
birthday    object
city        object
dtype: object
>>> ToDatetime().fit_transform(df["birthday"])
0   2024-01-29
Name: birthday, dtype: datetime64[...]
>>> ToDatetime().fit_transform(df["city"])
Traceback (most recent call last):
    ...
skrub._apply_to_cols.RejectColumn: Could not find a datetime format for column 'city'.

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
city                object
dtype: object


|ApplyToFrame| is instead used in cases where multiple columns should be transformed
at once. This is the case when the transformer is expecting multiple columns at
once, e.g., to perform dimensionality reduction:

>>> import numpy as np
>>> import pandas as pd
>>> df = pd.DataFrame(np.eye(4) * np.logspace(0, 3, 4), columns=list("abcd"))
>>> df
     a     b      c       d
0  1.0   0.0    0.0     0.0
1  0.0  10.0    0.0     0.0
2  0.0   0.0  100.0     0.0
3  0.0   0.0    0.0  1000.0

>>> from sklearn.decomposition import PCA
>>> from skrub import ApplyToFrame

Like with the other transformers described here, it is possible to limit the
transformations to a subset of columns:

>>> pca = ApplyToFrame(PCA(n_components=2), cols=["a", "b"])
>>> pca.fit_transform(df).round(2)
       c       d  pca0  pca1
0    0.0     0.0 -2.52  0.67
1    0.0     0.0  7.50  0.00
2  100.0     0.0 -2.49 -0.33
3    0.0  1000.0 -2.49 -0.33

By default, |ApplyToCols| and |ApplyToFrame| rename the transformed columns, and
remove the original features from the data. It is possible to rename the columns
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
        A  A_scaled      B  B_scaled
0 -10.0      -1.0    0.0      -1.0
1  10.0       1.0  100.0       1.0
