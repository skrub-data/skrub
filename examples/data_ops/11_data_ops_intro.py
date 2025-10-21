"""
Introduction to machine-learning pipelines with skrub DataOps
==============================================================

In this example, we show how we can use Skrub's
:ref:`DataOps <user_guide_data_ops_index>`
to build a machine learning pipeline that records all the operations involved in
pre-processing data and training a model. We will also show how to save the model,
load it back, and then use it to make predictions on new, unseen data.

This example is meant to be an introduction to Skrub DataOps, and as such it
will not cover all the features. Further examples in the gallery
:ref:`data_ops_examples_ref` will go into more detail on how to use Skrub DataOps
for more complex tasks.

.. currentmodule:: skrub

.. |fetch_employee_salaries| replace:: :func:`datasets.fetch_employee_salaries`
.. |TableReport| replace:: :class:`TableReport`
.. |var| replace:: :func:`var`
.. |skb.mark_as_X| replace:: :meth:`DataOp.skb.mark_as_X`
.. |skb.mark_as_y| replace:: :meth:`DataOp.skb.mark_as_y`
.. |TableVectorizer| replace:: :class:`TableVectorizer`
.. |skb.apply| replace:: :meth:`.skb.apply() <DataOp.skb.apply>`
.. |HistGradientBoostingRegressor| replace::
   :class:`~sklearn.ensemble.HistGradientBoostingRegressor`
.. |.skb.full_report()| replace:: :meth:`.skb.full_report() <DataOp.skb.full_report>`
.. |choose_float| replace:: :func:`choose_float`
.. |make_randomized_search| replace::
   :meth:`.skb.make_randomized_search <DataOp.skb.make_randomized_search>`

"""

# %%
# The data
# ---------
#
# We begin by loading the employee salaries dataset, which is a regression dataset
# that contains information about employees and their current annual salaries.
# By default, the |fetch_employee_salaries| function returns the training set.
# We will load the test set later, to evaluate our model on unseen data.

from skrub.datasets import fetch_employee_salaries

training_data = fetch_employee_salaries(split="train").employee_salaries

# %%
# We can take a look at the dataset using the |TableReport|.
# This dataset contains numerical, categorical, and datetime features. The column
# ``current_annual_salary`` is the target variable we want to predict.
#

import skrub

skrub.TableReport(training_data)
# %%
# Assembling our DataOps plan
# ----------------------------
#
# Our goal is to predict the ``current_annual_salary`` of employees based on their
# other features. We will use skrub's DataOps to combine both skrub and scikit-learn
# objects into a single DataOps plan, which will allow us to preprocess the data,
# train a model, and tune hyperparameters.
#
# We begin by defining a skrub |var|, which is the entry point for our DataOps plan.

data_var = skrub.var("data", training_data)

# %%
# Next, we define the initial features ``X`` and the target variable ``y``.
# We use the |skb.mark_as_X| and |skb.mark_as_y| methods to mark these variables
# in the DataOps plan. This allows skrub to properly split these objects into
# training and validation steps when executing cross-validation or hyperparameter
# tuning.

X = data_var.drop("current_annual_salary", axis=1).skb.mark_as_X()
y = data_var["current_annual_salary"].skb.mark_as_y()
# %%
# Our first step is to vectorize the features in ``X``. We will use the
# |TableVectorizer| to convert the categorical and numerical features into a
# numerical format that can be used by machine learning algorithms.
# We apply the vectorizer to ``X`` using the |skb.apply| method, which allows us to
# apply any scikit-learn compatible transformer to the skrub variable.

from skrub import TableVectorizer

vectorizer = TableVectorizer()

X_vec = X.skb.apply(vectorizer)
X_vec
# %%
# By clicking on ``Show graph``, we can see the DataOps plan that has been created:
# the plan shows the steps that have been applied to the data so far.
# Now that we have the vectorized features, we can proceed to train a model.
# We use a scikit-learn |HistGradientBoostingRegressor| to predict the target variable.
# We apply the model to the vectorized features using ``.skb.apply``, and pass
# ``y`` as the target variable.
# Note that the resulting ``predictor`` variable shows prediction results on the
# preview subsample, but the model will be properly fitted when we create the learner.

from sklearn.ensemble import HistGradientBoostingRegressor

hgb = HistGradientBoostingRegressor()

predictor = X_vec.skb.apply(hgb, y=y)
predictor

# %%
# Now that we have built our entire plan, we can explore it in more detail
# with the |.skb.full_report()| method::
#
#     predictions.skb.full_report()
#
# This produces a folder on disk rather than displaying inline in a notebook so
# we do not run it here. But you can
# `see the output here <../../_static/employee_salaries_report/index.html>`_.
#
# This method evaluates each step in
# the plan and shows detailed information about the operations that are being performed.

# %%
# Turning the DataOps plan into a learner, for later reuse
# ---------------------------------------------------------
#
# Now that we have defined the predictor, we can create a ``learner``, a
# standalone object that contains all the steps in the DataOps plan. We fit the
# learner, so that it can be used to make predictions on new data.

trained_learner = predictor.skb.make_learner(fitted=True)

# %%
# A big advantage of the learner is that it can be pickled and saved to disk,
# allowing us to reuse the trained model later without needing to retrain it.
# The learner contains all steps in the DataOps plan, including the fitted
# vectorizer and the trained model. We can save it using Python's ``pickle`` module.
# Here we use ``pickle.dumps`` to serialize the learner object into a byte string.

import pickle

saved_model = pickle.dumps(trained_learner)

# %%
# We can now load the saved model back into memory using ``pickle.loads``.
loaded_model = pickle.loads(saved_model)

# %%
# Now, we can make predictions on new data using the loaded model, by passing
# a dictionary with the skrub variable names as keys.
# We don't have to create a new variable, as this will be done internally by the
# learner.
# In fact, the ``learner`` is similar to a scikit-learn estimator, but rather
# than taking ``X`` and ``y`` as inputs, it takes a dictionary (the "environment")
# where each key corresponds to the name of a skrub variable in the plan (in this
# case, "data").
#
# We can now get the test set of the employee salaries dataset:
unseen_data = fetch_employee_salaries(split="test").employee_salaries

# %%
# Then, we can use the loaded model to make predictions on the unseen data by
# passing a dictionary with the variable name as the key.

predicted_values = loaded_model.predict({"data": unseen_data})
predicted_values

# %%
# We can also evaluate the model's performance using the `score` method, which
# uses the scikit-learn scoring function used by the predictor:
loaded_model.score({"data": unseen_data})

# %%
# Conclusion
# ----------
#
# In this example, we have briefly introduced the skrub DataOps and how they can
# be used to build powerful machine learning pipelines. We have shown how to preprocess
# data and train a model. We have also demonstrated how to save and load the trained
# model, and how to make predictions on new data.
#
# However, skrub DataOps are significantly more powerful than what we have shown here.
# For more advanced examples, see :ref:`data_ops_examples_ref`.
