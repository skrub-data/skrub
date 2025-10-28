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
>>> with open("learner.pkl", "wb") as f:
...     pickle.dump(learner, f)
>>> with open("learner.pkl", "rb") as f:
...     loaded_learner = pickle.load(f)
>>> loaded_learner.fit({"orders": new_orders_df})
SkrubLearner(data_op=<Apply TableVectorizer>)

See :ref:`sphx_glr_auto_examples_data_ops_15_use_case.py` for an example of how
to use the learner in a microservice.
