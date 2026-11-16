.. currentmodule:: skrub
.. _user_guide_data_ops_exporting:

Exporting the DataOps plan as a learner and reusing it
========================================================

DataOps are designed to build complex pipelines that can be reused on new, unseen
data in potentially different environments from where they were created. This can
be achieved by exporting the DataOps plan as a **learner**: the learner is special
transformer that is similar to a scikit-learn estimator, but that takes as input
the **environment** that should be used to execute the operations. The environment
is a dictionary of variables rather than a single design matrix
``X`` and a target array ``y``.

>>> import pandas as pd
>>> orders_df = pd.DataFrame(
...     {
...         "item": ["pen", "cup", "pen", "fork"],
...         "price": [1.5, None, 1.5, 2.2],
...         "qty": [1, 1, 2, 4],
...     }
... )
>>> import skrub
>>> from skrub import TableVectorizer
>>> orders = skrub.var("orders", orders_df)
>>> transformed_orders = orders.skb.apply(TableVectorizer())
>>> learner = transformed_orders.skb.make_learner()

The learner can be fitted as it is exported by setting ``fitted=True`` when
creating it with :meth:`.skb.make_learner() <DataOp.skb.make_learner>`.
This will fit the learner on the data used for previews when the variables are defined
(``orders_df`` in the case above):

>>> learner = transformed_orders.skb.make_learner(fitted=True)

Alternatively, the learner can be fitted on a different dataset by passing
the data to the ``fit()`` method:

>>> new_orders_df = pd.DataFrame(
...     {
...         "item": ["pen", "cup", "spoon"],
...         "price": [1.5, 2.0, 1.0   ],
...         "qty": [1, 2, 3],
...     }
... )
>>> learner.fit({"orders": new_orders_df})
SkrubLearner(data_op=<Apply TableVectorizer>)


The learner can be fitted and applied to new data
using the same methods as a scikit-learn estimator, such as ``fit()``,
``fit_transform()``, and ``predict()``.

The learner can be pickled and saved to disk, so that it can be reused later
or in a different environment:

>>> import pickle
>>> import io
>>> file = io.BytesIO()
>>> pickle.dump(learner, file)
>>> file.seek(0)
0
>>> loaded_learner = pickle.load(file)
>>> loaded_learner.fit({"orders": new_orders_df})
SkrubLearner(data_op=<Apply TableVectorizer>)

See :ref:`sphx_glr_auto_examples_02_data_ops_1150_use_case.py` for an example of how
to use the learner in a microservice.

.. _user_guide_data_ops_sklearn_interface:

Using a learner as a standard scikit-learn estimator
-----------------------------------------------------

A learner takes an environment dictionary as input, while many tools in the
scikit-learn ecosystem expect an estimator with the usual ``fit(X, y)`` interface.
When the DataOps plan needs only an ``X`` and possibly a ``y``, we can obtain such an
estimator by calling :meth:`SkrubLearner.with_sklearn_interface` on the learner.
The ``X`` and ``y`` passed to the estimator's methods are bound to the variables
named ``"X"`` and ``"y"`` (this can be changed with the parameters of
``with_sklearn_interface``):

>>> from sklearn.linear_model import LogisticRegression
>>> X = skrub.var("X")
>>> y = skrub.var("y")
>>> learner = X.skb.apply(LogisticRegression(), y=y).skb.make_learner()
>>> estimator = learner.with_sklearn_interface()
>>> estimator.fit(
...     [[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1]
... ).predict([[0.0], [3.0]])
array([0, 1])

The returned :class:`XySkrubLearner` can then be used in a scikit-learn
:class:`~sklearn.pipeline.Pipeline`, in
:func:`~sklearn.model_selection.cross_val_score`, or as the estimator of
another DataOp's :meth:`.skb.apply() <DataOp.skb.apply>`. Learners returned by
:meth:`.skb.make_grid_search() <DataOp.skb.make_grid_search>` and
:meth:`.skb.make_randomized_search() <DataOp.skb.make_randomized_search>` can be
wrapped in the same way.

See :ref:`example_with_sklearn_interface` for a more complete example and
:meth:`the reference documentation <SkrubLearner.with_sklearn_interface>` for
full details.
