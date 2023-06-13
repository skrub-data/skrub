=================================
Vision: where is skrub heading?
=================================

.. currentmodule:: skrub

Vision statement
=================

The goal of skrub is to facilitate building and deploying
machine-learning models on tables: `pandas <https://pandas.pydata.org>`__
dataframe, SQL databases...

|

Skrub is high-level, with a philosophy and an API matching that of
`scikit-learn <http://scikit-learn.org>`_. It strives to bridge the world
of databases to that of machine-learning, **enabling imperfect assembly and
representations of the data when it is noisy**, using the downstream
target to predict to guide assembly when possible (supervised learing for
data assembly).

In the long term, as skrub is built on higher-level APIs, it will make it
easier for data-scientist to use efficient database patterns and
backends.

Skrub seeks tradeoffs in terms of flexibility: its high-level APIs are by
construction restrictive compared to directly manipulating dataframes.
This is by design, as skrub does not aim to replace tools such as `Pandas
<https://pandas.pydata.org>`__, `Ibis <https://ibis-project.org>`__,
`DuckDB <https://duckdb.org/>`_.

To make things simpler, skrub uses defaults that are chosen empirically to
give good machine learning, even though these are sometimes heuristic, as
in the :class:`TableVectorizer`.


Roadmap
=========

In an open-source project, roadmaps can be whishful thinking: things
happen in an iterative way, often guided by the community.

We however decided to communicate on what we would like to do in the next
6 months to give a better idea of the vision.

From shorter term to longer term:

- Make the :class:`TableVectorizer` fast, robust, and easy to tune (in
  the sense of hyper-parameter tuning)

- Add a Join-aggregator object, to do feature augmentation on one-to-many
  correspondances

- fuzzy joining on datetime values

- Support polars?

- Support time series (eg in the aggregations)

- Interpolator join to join across multiple columns without exact
  correspondances in the keys

- Release (yes we are not planning to release very soon)

- Data namespaces, lazy data loading, out of core computing using
  database engines (eg duckdb)

- Join discovery to work in data lakes where the tables are not in a
  clean relational database

- Automatic feature synthesis in databases, building on the assembling
  features

