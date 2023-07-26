"""
Joining with keys across multiple columns
=========================================

Here we show how to combine data from different sources,
with a vocabulary not well normalized.

Joining is difficult: one entry on one side does not have
an exact match on the other side.

Joiner allows you to perform spatial joins.

The |fj| function enables to join tables without cleaning the data by
accounting for the label variations.

To illustrate, we will join data from the
`2022 World Happiness Report <https://worldhappiness.report/>`_, with tables
provided in `the World Bank open data platform <https://data.worldbank.org/>`_
in order to create a first prediction model.

Moreover, the |joiner| is a scikit-learn Transformer that makes it easy to
use such fuzzy joining multiple tables to bring in information in a
machine-learning pipeline. In particular, it enables tuning parameters of
|fj| to find the matches that maximize prediction accuracy.


.. |fj| replace:: :func:`~skrub.fuzzy_join`

.. |joiner| replace:: :func:`~skrub.Joiner`
"""
