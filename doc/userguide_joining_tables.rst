.. _userguide_joining_tables:


Joining Tables
--------------

``skrub`` features various objects that simplify combining information spread over multiple tables.

Approximate Join with ``fuzzy_join`` and ``Joiner``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``fuzzy_join`` function joins tables on approximate matches by vectorizing (embedding) the key columns in each table, then matching each row in the left table with its nearest neighbor in the right table according to the Euclidean distance between their embeddings.

The ``Joiner`` implements the same fuzzy join logic, but as a scikit-learn compatible transformer.

- Fuzzy joining with ``Joiner`` and ``fuzzy_join``
- ``AggJoiner`` and ``MultiAggJoiner``
- ``InterpolationJoiner``
