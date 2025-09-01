Joining external tables for machine learning
--------------------------------------------

Joining is straightforward for two tables because you only need to identify
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
