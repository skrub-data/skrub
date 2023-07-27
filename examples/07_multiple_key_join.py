"""
Joining with keys across multiple columns
=========================================

Joining is difficult: one entry on one side does not have
an exact match on the other side.

This problem becomes even more complex when multiple columns
are significant for the join.

Typically, this is the case in spatial joins: to join on
the closest location you need to take into account both
longitude and latitude.

|joiner| is a scikit-learn Transformer that allows you to perform
joining across multiple keys, and independantly of the data
type (numerical, string or mixed).


.. |fj| replace:: :func:`~skrub.fuzzy_join`

.. |joiner| replace:: :func:`~skrub.Joiner`
"""

from skrub.datasets import fetch_road_safety

data = fetch_road_safety()

X = data.X

X.head()

X[["Longitude", "Latitude"]]
