.. currentmodule:: skrub

.. _user_guide_data_ops_intro:


Basics of DataOps: the DataOps plan, variables, and learners
===============================================================

**DataOps** are special objects that encapsulate operations on data (such as
applying operators, or calling methods) to record the parameters so that they
can later be replayed on new data. DataOps objects can be combined into a
DataOps plan, which is a directed acyclic graph (DAG) of operations.

DataOps have a ``.skb`` attribute that provides access to the DataOps namespace,
which contains methods for evaluating the DataOps plan, exporting the plan as a
**learner**, and various other utilities. Any other operation on a DataOp that is
not part of the DataOps namespace is instead applied to the underlying data: this
allows, for example, to make use of Pandas or Polars methods if the DataOp is
encapsulating a DataFrame or Series.

The entry point of any DataOps plan is :class:`~skrub.var`,
a **variable**: a variable is an input to
our machine learning pipeline, such as a table of data, a target array, or more
generic data such as paths to files, or timestamps.

Variables can be combined using operators and function calls to build more
complex DataOps plans. The plan is constructed implicitly as we apply these
operations, rather than by specifying an explicit list of transformations.

At any point in the DataOps plan, we can export the resulting computation graph
as a **learner** with :meth:`~skrub.DataOp.skb.make_learner()`. A learner is a
special object akin to a scikit-learn estimator, but that takes as input a
dictionary of variables rather than a single design matrix ``X`` and a target array
``y``.
