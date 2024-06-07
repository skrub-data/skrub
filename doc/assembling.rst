
.. _assembling:

====================================
Assembling: joining multiple tables
====================================

.. currentmodule:: skrub

Assembling is the process of collecting and joining together tables.
Good analytics requires including as much information as possible,
often from different sources.

skrub allows you to join tables on keys of different types
(string, numerical, datetime) with imprecise correspondence.

Fuzzy joining tables
---------------------

Joining two dataframes can be hard as the corresponding keys may be different.

:func:`fuzzy_join` uses similarities in entries to join tables on one or more
related columns. Furthermore, it chooses the type of fuzzy matching used based
on the column type (string, numerical or datetime). It also outputs a similarity
score, to single out bad matches, so that they can be dropped or replaced.

In sum, equivalent to :func:`pandas.merge`, the :func:`fuzzy_join`
has no need for pre-cleaning.


Joining external tables for machine learning
--------------------------------------------

Joining is straigthforward for two tables because you only need to identify
the common key.

In addition, skrub also enable more advanced analysis:

- :class:`Joiner`: fuzzy-joins an external table using a scikit-learn
  transformer, which can be used in a scikit-learn :class:`~sklearn.pipeline.Pipeline`.
  Pipelines are useful for cross-validation and hyper-parameter search, but also
  for model deployment.

- :class:`AggJoiner`: instead of performing 1:1 joins like :class:`Joiner`, :class:`AggJoiner`
  aggregates the external table first, then joins it on the main table.
  Alternatively, it can aggregate the main table and then join it back onto itself.

- :class:`AggTarget`: in some settings, one can derive powerful features from
  the target `y` itself. AggTarget aggregates the target without risking data
  leakage, then joins the result back on the main table, similar to AggJoiner.

- :class:`MultiAggJoiner`: extension of the :class:`AggJoiner` that joins multiple
  auxiliary tables onto the main table.


Column selection inside a pipeline
----------------------------------

Besides joins, another common operation on a dataframe is to select a subset of its columns (also known as a projection).
We sometimes need to perform such a selection in the middle of a pipeline, for example if we need a column for a join (with :class:`Joiner`), but in a subsequent step we want to drop that column before fitting an estimator.

skrub provides transformers to perform such an operation:

- :class:`SelectCols` allows specifying the columns we want to keep.
- Conversely :class:`DropCols` allows specifying the columns we want to discard.

Going further: embeddings for better analytics
----------------------------------------------

Data collection comes before joining, but is also an
essential process of table assembling.
Although many datasets are available on the internet, it is not
always easy to find the right one for your analysis.

skrub has some very helpful methods that gives you easy
access to embeddings, or vectorial representations of an entity,
of all common entities from Wikipedia.
You can use :func:`datasets.get_ken_embeddings` to search for the right
embeddings and download them.

Other methods, such as :func:`datasets.fetch_world_bank_indicator` to
fetch data of a World Bank indicator can also help you retrieve
useful data that will be joined to another table.
