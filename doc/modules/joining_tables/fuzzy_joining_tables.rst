Fuzzy joining tables
---------------------

Joining two dataframes can be hard as the corresponding keys may be different.

:func:`fuzzy_join` uses similarities in entries to join tables on one or more
related columns. Furthermore, it chooses the type of fuzzy matching used based
on the column type (string, numerical or datetime). It also outputs a similarity
score, to single out bad matches, so that they can be dropped or replaced.

In sum, equivalent to :func:`pandas.merge`, the :func:`fuzzy_join`
has no need for pre-cleaning.
