.. |ApplyToCols| replace:: :class:`~skrub.ApplyToCols`
.. |MissingIndicator| replace:: :class:`~sklearn.impute.MissingIndicator`
.. |SimpleImputer| replace:: :class:`~sklearn.impute.SimpleImputer`

.. _howto_preserve_missingness:

How to retain missing-value information when imputing data
=========================================================

Replacing missing values with an imputed value can discard useful information:
the fact that a value was missing may itself help a predictive model. Instead
of dropping a column with missing data, use a |MissingIndicator| to retain a
boolean feature recording which rows originally had missing values.

The :func:`~skrub.selectors.has_nulls` selector finds columns containing
missing values. Combined with |ApplyToCols|, it lets us add indicators only
for those columns, while keeping the remaining columns unchanged.

In this example, ``measurement`` contains missing values and ``price``
does not:

>>> import pandas as pd
>>> from sklearn.impute import MissingIndicator, SimpleImputer
>>> from sklearn.pipeline import make_pipeline
>>> from skrub import ApplyToCols, selectors as s
>>> df = pd.DataFrame({
...     "measurement": [1.0, None, 3.0, None, 5.0],
...     "price": [10.0, 20.0, 30.0, 40.0, 50.0],
... })

First create an indicator while retaining the original columns. Then fill the
missing numeric values. Applying the imputer *after* the indicator ensures that
the indicator records missingness **before** imputation:

>>> indicator = ApplyToCols(
...     MissingIndicator(), cols=s.has_nulls(), keep_original=True
... )
>>> imputer = ApplyToCols(SimpleImputer(), cols=s.has_nulls())
>>> pipeline = make_pipeline(indicator, imputer)
>>> result = pipeline.fit_transform(df)

The generated ``missingindicator_measurement`` column is ``True`` exactly
where the original measurement was missing. The original numeric column is
retained and imputed:

>>> result["missingindicator_measurement"].tolist()
[False, True, False, True, False]
>>> int(result["measurement"].isna().sum())
0
>>> result["price"].tolist()
[10.0, 20.0, 30.0, 40.0, 50.0]


The fitted pipeline also adds indicators for new rows. The selected input
columns are unchanged after fitting, even when the missingness pattern changes:

>>> new_rows = pd.DataFrame({
...     "measurement": [None, 8.0],
...     "price": [15.0, 25.0],
... })
>>> transformed = pipeline.transform(new_rows)
>>> transformed["missingindicator_measurement"].tolist()
[True, False]
>>> int(transformed["measurement"].isna().sum())
0

To select *only* columns whose missing fraction exceeds a threshold, use
``s.has_nulls(proportion=0.4)`` instead. For example, a threshold of
``0.4`` selects columns with **more than 40%** missing values. The default
``s.has_nulls()`` selects columns with any missing values.

.. note::

   The selector is evaluated when the pipeline is fitted. For new rows passed
   to ``pipeline.transform()``, the fitted set of selected columns is
   reused. ``SimpleImputer`` in this example is intended for numeric columns;
   choose an appropriate imputer for categorical or datetime features.
