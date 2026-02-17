.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |Cleaner| replace:: :class:`~skrub.Cleaner`
.. |DropUninformative| replace:: :class:`~skrub.DropUninformative`
.. |DatetimeEncoder| replace:: :class:`~skrub.DatetimeEncoder`
.. |StringEncoder| replace:: :class:`~skrub.StringEncoder`
.. |OneHotEncoder| replace:: :class:`~sklearn.preprocessing.OneHotEncoder`
.. |OrdinalEncoder| replace:: :class:`~sklearn.preprocessing.OrdinalEncoder`
.. |TextEncoder| replace:: :class:`~skrub.TextEncoder`
.. |ApplyOnEachCol| replace:: :class:`~skrub.ApplyOnEachCol`
.. |ToCategorical| replace:: :class:`~skrub.ToCategorical`

.. _user_guide_table_vectorizer:

Transforming a table into an array of numeric features: |TableVectorizer|
-------------------------------------------------------------------------

In tabular machine learning pipelines, practitioners often convert categorical
features to numeric features using various encodings (|OneHotEncoder|, |OrdinalEncoder|,
etc.).

The objective of the |TableVectorizer| is to take any dataframe as input, and
produce as output a feature-engineered version of the dataframe.

Initially, the |TableVectorizer| parses the data type of each column and maps each
column to an encoder, in order to produce numeric features for machine learning
models.

Parsing is handled internally by running a |Cleaner| on the input data.
Note that in this  case numeric values are always converted to ``float32``
(whereas the default |Cleaner| behavior is to keep the original datatype): this
is to ensure that the numeric dtype (including that of the missing values) is
consistent for the downstream methods. For most applications, ``float32`` has a
sufficient precision, and reduces the memory footprint of the resulting features.

The same parameters used for the |Cleaner| can also be set when creating the
|TableVectorizer|: this includes parameters for |DropUninformative|
(``drop_null_fraction`` etc.), and a ``datetime_format`` parameter for the
datetime parsing step.


After detecting the datatypes, the |TableVectorizer| maps columns to one of
four groups depending either on the datatype, and the number of unique values
for categorical/string columns

The default transformers used by the |TableVectorizer| for each column category
are the following:

- **High-cardinality categorical columns**: |StringEncoder|
- **Low-cardinality categorical columns**: scikit-learn |OneHotEncoder|
- **Numeric columns**: "passthrough" (no transformation)
- **Datetime columns**: |DatetimeEncoder|

**High cardinality** categorical columns are those with more than 40 unique values,
while all other categorical columns are considered **low cardinality**: the
threshold can be changed by setting the ``cardinality_threshold`` parameter of
|TableVectorizer|, or by changing the configuration parameter with the same name
using :func:`~skrub.set_config`.

To change the encoder or alter default parameters, instantiate an encoder and pass
it to |TableVectorizer|.

>>> from skrub import TableVectorizer, DatetimeEncoder, TextEncoder, SquashingScaler

>>> datetime_enc = DatetimeEncoder(periodic_encoding="circular")
>>> text_enc = TextEncoder()
>>> num_enc = SquashingScaler()
>>> table_vec = TableVectorizer(datetime=datetime_enc, high_cardinality=text_enc, numeric=num_enc)
>>> table_vec
TableVectorizer(datetime=DatetimeEncoder(periodic_encoding='circular'),
                high_cardinality=TextEncoder(), numeric=SquashingScaler())


Besides the transformers provided by skrub, the |TableVectorizer| can also take
user-specified transformers that are applied to given columns.

>>> from sklearn.preprocessing import OrdinalEncoder
>>> import pandas as pd
>>> encoder = OrdinalEncoder()
>>> df = pd.DataFrame({
...     "values": ["A", "B", "C"]
... })

We define the list of column-specific transformers:

>>> specific_transformers=[(encoder, ["values"])]

We can then encode the result:

>>> TableVectorizer(specific_transformers=specific_transformers).fit_transform(df)
   values
0     0.0
1     1.0
2     2.0

Note that the columns specified in ``specific_transformers`` are passed to the
transformer without any modification, which means that the transformer must be
able to handle the content of the column on its own.

If you need to define complex transformers to pass to a single instance of
|TableVectorizer|, consider using the :ref:`skrub Data Ops <user_guide_data_ops_index>`,
|ApplyOnEachCol|, or the :ref:`skrub selectors <user_guide_selectors>` instead, as
they are more versatile and allow a higher degree
of control over which operations are applied to which columns.

The |TableVectorizer| is used in :ref:`example_encodings`, while the
docstring of the class provides more details on the parameters and usage, as well
as various examples.

Numeric strings and categorical encoding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, columns that contain only numeric strings (e.g. ``["1", "2", "3"]``)
are parsed as numeric features by the |TableVectorizer|. The recommended way to
treat such values as categorical (e.g. IDs or codes) is to convert the column
to pandas' ``category`` dtype using |ToCategorical| with |ApplyOnEachCol| before
vectorizing, rather than relying on keeping them as strings.

Default behavior: numeric strings are parsed as a single numeric column (feature
names are not one-hot encoded):

>>> import pandas as pd
>>> from skrub import TableVectorizer
>>> df = pd.DataFrame({"c": ["1", "2", "3"]})
>>> tv = TableVectorizer().fit(df)
>>> list(map(str, sorted(tv.get_feature_names_out())))
['c']
>>> tv = TableVectorizer()
>>> tv.fit_transform(df)
c
0  1.0
1  2.0
2  3.0

With |ToCategorical| and |ApplyOnEachCol|, the column is treated as categorical
and produces one-hot encoded feature names:

>>> from skrub import ApplyOnEachCol, TableVectorizer, ToCategorical
>>> from sklearn.pipeline import make_pipeline
>>> pipe = make_pipeline(
...     ApplyOnEachCol(ToCategorical(), cols=["c"]),
...     TableVectorizer(),
... )
>>> pipe.fit(df)
Pipeline(steps=[('ApplyOnEachCol', ...), ('tablevectorizer', ...)])
>>> list(map(str, sorted(pipe.named_steps["tablevectorizer"].get_feature_names_out())))
['c_1', 'c_2', 'c_3']
>>> pipe.fit_transform(df)
   c_1  c_2  c_3
0  1.0  0.0  0.0
1  0.0  1.0  0.0
2  0.0  0.0  1.0
