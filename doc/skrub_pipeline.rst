.. _skrub_pipeline:

===================================================
Skrub Pipeline: flexible machine learning pipelines
===================================================

.. currentmodule:: skrub

Skrub provides an easy way to build complex, flexible machine-learning
pipelines. It solves several problems that are not easily addressed with the
standard scikit-learn tools such as the ``Pipeline`` and ``ColumnTransformer``.

**Multiple tables:** we have several tables of different shapes (for example,
we may have "Customers", "Orders" and "Products" tables). But scikit-learn
estimators expect a single design matrix ``X`` and array of targets ``y`` with one row
per observation.

**DataFrame wrangling:** we need to easily perform typical dataframe operations
such as projections, joins and aggregations leveraging the powerful APIs of
``pandas`` or ``polars``.

**Iterative development:** we want to build a pipeline step by step, while
inspecting the intermediate results so that the feedback loop is short and
errors are discovered early.

**Hyperparameter tuning:** many choices such as estimators, hyperparameters,
even the architecture of the pipeline can be informed by validation scores.
Specifying the grid of hyperparameters separately from the model (as in
``GridSearchCV``) is very difficult for complex pipelines.

What is the difference with scikit-learn :class:`~sklearn.pipelines.Pipelines`?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scikit-learn pipelines represent a linear sequence of transformations on one
table with a fixed number of rows.

.. image:: ../../_static/sklearn_pipeline.svg
    :width: 500

Skrub expressions, on the other hand, can manipulate any number of variables.
The transformation they perform is not a linear sequence but any Directed
Acyclic Graph of computations.

.. image:: ../../_static/skrub_expressions.svg


Skrub expressions
=================
