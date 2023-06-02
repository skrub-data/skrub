====================================
Assembling: joining multiple tables
====================================

.. currentmodule:: skrub

Assembling is the process of collecting and joining together tables. 
Good analytics requires including as much information as possible,
often from different sources.
skrub allows you to join tables on keys of different types
(string, numerical, datetime) with imprecise correspondance.

Fuzzy joining tables
---------------------

Joining two dataframes can be hard as the corresponding keys may be different.

The :func:`fuzzy_join` uses similarities in entries to join tables on one or more
related columns. Furthermore, it adapts the fuzzy matching based on the
column type (string, numerical or datetime).
Using the similarity score, bad matches are easily singled out and can be
dropped or replaced.

In sum, equivalent to pandas.merge, the :func:`fuzzy_join` 
has no need for pre-cleaning.


Feature augmentation for machine learning
-----------------------------------------
Joining is pretty straigthforward for two tables: you only need to identify
the common key.
However, for more complex analysis, merging multiple tables is necessary.
skrub provides the :class:`FeatureAugmenter` as a convenient solution:
multiple fuzzy joins can be performed at the same time, given a set of 
input tables and key columns.

An advantage is the scikit-learn compatibility of this class:
easily introduced into machine learning pipelines.


Going further: embeddings for better analytics
----------------------------------------------

