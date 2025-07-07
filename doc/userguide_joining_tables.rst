.. _userguide_joining_tables:
.. |Joiner| replace:: :class:`~skrub.Joiner`
.. |fuzzy_join| replace:: :func:`~skrub.fuzzy_join`
.. |AggJoiner| replace:: :class:`~skrub.AggJoiner`
.. |MultiAggJoiner| replace:: :class:`~skrub.MultiAggJoiner`
.. |InterpolationJoiner| replace:: :class:`~skrub.InterpolationJoiner`


Merging tables in imperfect conditions with the ``skrub`` Joiners
--------------

``skrub`` features various objects that simplify merging tables in imperfect conditions.
The |Joiner| joins tables by matching rows based on a key column, allowing for approximate matches and fuzzy logic.
The |fuzzy_join| function performs a similar operation, but as a standalone function rather than a transformer.
The |AggJoiner| and |MultiAggJoiner| perform aggregations on the right table before joining, allowing for more complex merging operations.
The |InterpolationJoiner| performs a join based on interpolating values from the right table to the left table, which is useful for time series data or when the right table contains values that need to be estimated for the left table's keys.

Approximate Join with |fuzzy_join| and |Joiner|
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The |fuzzy_join| function joins tables on approximate matches by vectorizing (embedding)
the key columns in each table, then matching each row in the left table with its nearest
neighbor in the right table according to the Euclidean distance between their embeddings.
The |Joiner| implements the same fuzzy join logic, but as a scikit-learn compatible transformer.

- |AggJoiner| and |MultiAggJoiner|
- |InterpolationJoiner|
