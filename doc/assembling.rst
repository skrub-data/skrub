
.. _assembling:

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
related columns. Furthermore, it choose the type of fuzzy matching used
based on the column type (string, numerical or datetime).
It also outputs a similarity score, to single out bad matches, so that they can be
dropped or replaced.

In sum, equivalent to :func:`pandas.merge`, the :func:`fuzzy_join` 
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
