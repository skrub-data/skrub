.. _dataops_vs_pipeline:

How do Skrub DataOps differ from sklearn.pipeline.Pipeline?
===========================================================

Scikit-learn pipelines represent a linear sequence of transformations on one
table with a fixed number of rows.

.. image:: _static/sklearn_pipeline.svg
    :width: 500

Skrub DataOps, on the other hand, can manipulate any number of variables.
The transformation they perform is not a linear sequence but any Directed
Acyclic Graph of computations.
