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

Joining two dataframes is hard as the corresponding keys are often different.

The :func:`fuzzy_join` uses similarity scores to join tables on one or more
related columns. Furthermore, it adapts the fuzzy matching based on the
column type (string, numerical or datetime).
In sum; equivalent to pandas.merge, but no need for pre-cleaning.


Feature augmentation for machine learning
-----------------------------------------
WIP

Going further: embeddings for better analytics
----------------------------------------------
