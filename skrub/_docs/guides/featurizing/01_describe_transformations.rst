.. |TableVectorizer| replace:: :class:`~skrub.TableVectorizer`
.. |Cleaner| replace:: :class:`~skrub.Cleaner`
.. |DatetimeEncoder| replace:: :class:`~skrub.DatetimeEncoder`
.. |OneHotEncoder| replace:: :class:`~sklearn.preprocessing.OneHotEncoder`

.. _user_guide_featurizing_describe_transformations:

How to display how the |TableVectorizer| modified a dataframe
=============================================================

Both the |TableVectorizer| and the |Cleaner| modify a given dataframe in various
ways, from converting dtypes, to cleaning formats, to removing columns. In many
situations, knowing which columns have been modified and how is important to
understand what is happening in a pipeline.

Because all of this happens automatically, it can be useful to get a summary of
what was actually done to each column. Once a |TableVectorizer| or |Cleaner| is
fitted, the ``describe_transformations`` method returns a human-readable report
listing:

- the **preprocessing** steps and the columns they were applied to;
- (for the |TableVectorizer|) the **main transformers** (one per column kind) and
  the columns they handled.

Let us fit a |TableVectorizer| on a small employee dataframe that contains a
numeric column, a date stored as a string, a percentage stored as a string, and
two string columns:

>>> import pandas as pd
>>> from skrub import TableVectorizer

>>> df = pd.DataFrame({
...     "department": ["sales", "sales", "engineering", "hr", "hr"],
...     "employee": ["ann smith", "bob jones", "carla diaz", "dan wu", "eve brown"],
...     "salary": [55000.0, 61000.0, 72000.0, 51000.0, 59000.0],
...     "hire_date": ["2020-01-15", "2019-06-30", "2021-03-22", "2018-11-05", "2022-07-19"],
...     "bonus_pct": ["5%", "7.5%", "10%", "4.5%", "6%"],
... })
>>> vectorizer = TableVectorizer().fit(df)

This prints the following report:

>>> print(vectorizer.describe_transformations()) # doctest: +SKIP

.. code-block::

   Preprocessors
   =============
   Null values cleaned (4 columns):
       - department
       - employee
       - hire_date
       - bonus_pct

   Processors by type
   ==================
   PassThrough (numeric - 1 columns):
       - salary
   DatetimeEncoder (datetime - 1 columns):
       - hire_date
   OneHotEncoder (low_cardinality - 3 columns):
       - department
       - employee
       - bonus_pct
   No high_cardinality columns have been detected.

Reading the report
------------------

- ``salary`` is already numeric, so it is passed through unchanged
  (``PassThrough``).
- ``hire_date`` contains dates stored as strings; the |TableVectorizer| parsed
  them and applied the |DatetimeEncoder| to extract features such as the year,
  month and day.
- ``department``, ``employee`` and ``bonus_pct`` are string columns with few
  unique values, so they are handled by the |OneHotEncoder|.

The ``max_cols`` parameter limits how many columns are listed per transformer;
any overflow is represented by ``...``:

.. code-block::

    print(vectorizer.describe_transformations(max_cols=2))

For a programmatic view of the same information, you can inspect the fitted
attributes of the |TableVectorizer|: ``column_to_kind_`` maps each input column
to its kind, ``transformers_`` maps each column to the fitted transformer that
was applied to it, and ``all_processing_steps_`` lists every processing step
(including preprocessing and the final cast to ``float32``) applied to each
column:

>>> vectorizer.column_to_kind_
{'salary': 'numeric',
 'hire_date': 'datetime',
 'department': 'low_cardinality',
 'employee': 'low_cardinality',
 'bonus_pct': 'low_cardinality'}

>>> vectorizer.transformers_["hire_date"]
DatetimeEncoder()

>>> vectorizer.all_processing_steps_["hire_date"]
[CleanNullStrings(),
 DropUninformative(),
 ToDatetime(),
 DatetimeEncoder(),
 {'hire_date_day': ToFloat(), 'hire_date_month': ToFloat(), ...}]

The |Cleaner| also provides the ``.all_processing_steps_`` method.
