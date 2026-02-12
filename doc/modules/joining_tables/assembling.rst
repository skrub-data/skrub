.. currentmodule:: skrub

Assembling: joining multiple tables
===================================

Assembling is the process of collecting and joining together tables. Good analytics
requires including as much information as possible, often from different sources.

Skrub allows you to join tables on keys of different types (string, numerical,
datetime) with imprecise correspondence.



Joining external tables for machine learning
--------------------------------------------

Joining is straightforward for two tables because you only need to identify
the common key.

In addition, skrub also enable more advanced analysis:

- :class:`Joiner`: fuzzy-joins an external table using a scikit-learn
  transformer, which can be used in a scikit-learn :class:`~sklearn.pipeline.Pipeline`.
  Pipelines are useful for cross-validation and hyper-parameter search, but also
  for model deployment.

- :class:`AggJoiner`: instead of performing 1:1 joins like :class:`Joiner`,
  :class:`AggJoiner`
  aggregates the external table first, then joins it on the main table.
  Alternatively, it can aggregate the main table and then join it back onto itself.

- :class:`AggTarget`: in some settings, one can derive powerful features from
  the target ``y`` itself. AggTarget aggregates the target without risking data
  leakage, then joins the result back on the main table, similar to AggJoiner.

- :class:`MultiAggJoiner`: extension of the :class:`AggJoiner` that joins multiple
  auxiliary tables onto the main table.

Fuzzy joining tables
---------------------

Joining two dataframes can be hard as the corresponding keys may be different.

:func:`~skrub.fuzzy_join` uses similarities in entries to join tables on one or more
related columns. Furthermore, it chooses the type of fuzzy matching used based
on the column type (string, numeric or datetime). It also outputs a similarity
score, to single out bad matches, so that they can be dropped or replaced.

In sum, equivalent to :func:`pandas.merge`, the :func:`fuzzy_join`
has no need for pre-cleaning.


Using the :class:`InterpolationJoiner` to join tables using ML predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :class:`InterpolationJoiner` is a transformer that performs an operation similar
to that of a regular equi-join, but that can handle the presence of missing rows
in the right table (the table to be added). This is done by estimating the value
that the missing rows would have by training a machine learning model on the data
we have access to.

This transformer is explored in more detail in :ref:`this example <sphx_glr_auto_examples_0080_interpolation_join.py>`.
