"""

.. currentmodule:: skrub

.. _example_tuning_pipelines:

Hyperparameter tuning with DataOps
==================================

A machine-learning pipeline typically contains values or choices which
may influence its prediction performance, such as hyperparameters (e.g., the
regularization parameter ``alpha`` of a :class:`~sklearn.linear_model.RidgeClassifier`,
the ``learning_rate`` of a :class:`~sklearn.ensemble.HistGradientBoostingClassifier`),
which estimator to use (e.g., ``RidgeClassifier`` or
``HistGradientBoostingClassifier``),
or which steps to include (e.g., should we join a table to bring additional information
or not).

We want to tune these choices by trying several options and keeping those that
give the best performance on a validation set.

Skrub :ref:`DataOps <user_guide_data_ops_index>` provide a convenient way to specify
the range of possible values by inserting them directly in place of the actual
value. For example, we can write:
"""

# %%
from sklearn.linear_model import RidgeClassifier

import skrub

RidgeClassifier(alpha=skrub.choose_from([0.1, 1.0, 10.0], name="α"))

# %%
# instead of:

RidgeClassifier(alpha=1.0)

# %%
# Skrub then inspects our DataOps plan to discover all the places where we used objects
# like :func:`~skrub.choose_from()` and builds a grid of hyperparameters for us.
#
# We will illustrate hyperparameter tuning on the "toxicity" dataset. This
# dataset contains 1,000 texts and the task is to predict if they are
# flagged as being toxic or not.
#
# We start from a very simple pipeline without any hyperparameters.

# %%
from sklearn.ensemble import HistGradientBoostingClassifier

import skrub
import skrub.datasets

data = skrub.datasets.fetch_toxicity().toxicity

# This dataset is sorted -- all toxic tweets appear first, so we shuffle it
data = data.sample(frac=1.0, random_state=1)

texts = data[["text"]]
labels = data["is_toxic"]

# %%
# We mark the ``texts`` column as the input variable and the ``labels`` column as
# the target variable.
#
# See `the previous example <10_data_ops_intro.html>`_
# for a more detailed explanation
# of :func:`skrub.X` and :func:`skrub.y`.
#
# We then encode the text with a :class:`~skrub.MinHashEncoder` and fit a
# :class:`~sklearn.ensemble.HistGradientBoostingClassifier` on the resulting features.

# %%
X = skrub.X(texts)
X

# %%
y = skrub.y(labels)
y

# %%
pred = X.skb.apply(skrub.MinHashEncoder()).skb.apply(
    HistGradientBoostingClassifier(), y=y
)
pred.skb.cross_validate(n_jobs=4)["test_score"]

# %%
# In this example, we will focus on the ``n_components`` of the
# ``MinHashEncoder`` and the ``learning_rate`` of the ``HistGradientBoostingClassifier``
# to illustrate the choices objects.
#
# When we use a scikit-learn hyperparameter-tuner like
# :class:`~sklearn.model_selection.GridSearchCV` or
# :class:`~sklearn.model_selection.RandomizedSearchCV`, we need to specify a grid of
# hyperparameters separately from the estimator, with something similar to
# ``GridSearchCV(my_pipeline, param_grid={"encoder__n_components: [5, 10, 20]"})``.
#
# Instead, within a skrub DataOps plan we can use
# ``skrub.choose_from(...)`` directly where the actual value
# would normally go. Skrub then takes care of constructing the
# :class:`~sklearn.model_selection.GridSearchCV`'s parameter grid for us.
#
# Note that :func:`skrub.choose_float()` and :func:`skrub.choose_int()` can be given a
# ``log`` argument to sample in log scale, and that it is possible to specify the
# number of steps with the ``n_steps`` argument.

# %%
X, y = skrub.X(texts), skrub.y(labels)

encoder = skrub.MinHashEncoder(
    n_components=skrub.choose_int(5, 15, n_steps=5, name="N components")
)
classifier = HistGradientBoostingClassifier(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="lr")
)
pred = X.skb.apply(encoder).skb.apply(classifier, y=y)

# %%
# From here, the ``pred`` DataOp can be used to perform hyperparameter search with
# ``.skb.make_grid_search()`` or ``.skb.make_randomized_search()``. They accept
# the same arguments as their scikit-learn counterparts (e.g., ``scoring``, ``cv``,
# ``n_jobs``). Also, like ``.skb.make_learner()``, they accept a ``fitted``
# argument: if ``fitted=True``, the search is fitted on the data we provided
# when initializing our pipeline's variables.

search = pred.skb.make_randomized_search(
    n_iter=8, n_jobs=4, random_state=1, fitted=True
)
search.results_

# %%
# If the plotly library is installed, we can visualize the results of the
# hyperparameter search with :func:`~skrub.ParamSearch.plot_results`.
# In the plot below, each line represents a combination of hyperparameters (in
# this case, only ``N components`` and ``learning rate``), and each column of
# points represents either a hyperparameter or the score of a given
# combination of hyperparameters.
#
# The color of the line represents the score of the combination of hyperparameters.
# The plot is interactive, and you can select only a subset of the
# hyperparameters to visualize by dragging the mouse over each column to select
# the desired range.
#
# This is particularly useful when there are many combinations of hyperparameters,
# and we want to understand which hyperparameters have the largest
# impact on the score.

search.plot_results()
# %%
# Finally, we can retrieve the best learner from the search results, and save it
# to disk. This learner will contain the best hyperparameter configuration
# found during the search, and can be used to make predictions on new data.

import pickle

best_learner = search.best_learner_
saved_model = pickle.dumps(best_learner)

# %%
# Default choice values
# ---------------------
#
# The goal of using the different ``choose_*`` functions is to tune choices on
# validation metrics with randomized or grid search. However, even when our
# expression contains such choices we can still use it without tuning, for
# example in previews or to get a quick first result before spending the
# computation time to run the search. When we use :meth:`.skb.make_learner()
# <DataOp.skb.make_learner>`, we get a pipeline that does not perform any tuning
# and uses those default values. This default pipeline is used for
# :meth:`.skb.eval() <DataOp.skb.eval>`.
#
# We can control what should be the default value for each choice. For
# :func:`choose_int`, :func:`choose_float` and :func:`choose_bool`, we can use
# the ``default`` parameter. For :func:`choose_from`, the default is the first
# item from the list or dict of outcomes we provide. For :func:`optional`, we
# can pass ``default=None`` to force the default to be the alternative
# outcome, ``None``.
#
# When we do not set an explicit default, skrub picks one for depending on the
# kind of choice, as detailed in :ref:`this table<choice-defaults-table>` in the
# User Guide.

# %%
# As mentioned we can control the default value:

# %%
skrub.choose_float(1.0, 100.0, default=12.0).default()

# %%
# Choices can appear in many places
# ---------------------------------
#
# Choices are not limited to selecting estimator hyperparameters. They can also be
# used to choose between different estimators, or in place of any value used in
# our pipeline.
#
# For example, here we pass a choice to pandas DataFrame's ``assign`` method.
# We want to add a feature that captures the length of the text, but we are not
# sure if it is better to count length in characters or in words. We do not
# want to add both because it would be redundant. We can add a column to the
# dataframe, which will be chosen among the length in characters or the length
# in words:

# %%
X, y = skrub.X(texts), skrub.y(labels)

X.assign(
    length=skrub.choose_from(
        {"words": X["text"].str.count(r"\b\w+\b"), "chars": X["text"].str.len()},
        name="length",
    )
)

# %%
# ``choose_from`` can be given a dictionary if we want to provide
# names for the individual outcomes, or a list, when names are not needed:
# ``choose_from([1, 100], name='N')``,
# ``choose_from({'small': 1, 'big': 100}, name='N')``.
#
# Choices can be nested arbitrarily. For example, here we want to choose
# between 2 possible encoder types: the ``MinHashEncoder`` or the
# ``StringEncoder``. Each of the possible outcomes contains a choice itself:
# the number of components.

# %%
X, y = skrub.X(texts), skrub.y(labels)

n_components = skrub.choose_int(5, 15, name="N components")

encoder = skrub.choose_from(
    {
        "minhash": skrub.MinHashEncoder(n_components=n_components),
        "lse": skrub.StringEncoder(n_components=n_components),
    },
    name="encoder",
)
X.skb.apply(encoder, cols="text")

# %%
# In a similar vein, we might want to choose between a HGB classifier and a Ridge
# classifier, each with its own set of hyperparameters.
# We can then define a choice for the classifier and a choice for the
# hyperparameters of each classifier.

# %%
from sklearn.linear_model import RidgeClassifier

hgb = HistGradientBoostingClassifier(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="lr")
)
ridge = RidgeClassifier(alpha=skrub.choose_float(0.01, 100, log=True, name="α"))
classifier = skrub.choose_from({"hgb": hgb, "ridge": ridge}, name="classifier")
pred = X.skb.apply(encoder).skb.apply(classifier, y=y)
print(pred.skb.describe_param_grid())

# %%
search = pred.skb.make_randomized_search(
    n_iter=16, n_jobs=4, random_state=1, fitted=True
)
search.plot_results()

# %%
# Now that we have a more complex plan, we can draw more conclusions from the
# parallel coordinate plot. For example, we can see that the
# ``HistGradientBoostingClassifier``
# performs better than the ``RidgeClassifier`` in most cases, that the ``StringEncoder``
# outperforms the ``MinHashEncoder``, and that the choice of the additional ``length``
# feature does not have a significant impact on the score.

# %%
# In this example, we've seen how to use skrub's ``choose_from`` objects to tune
# hyperparameters, choose optional configurations, and nest choices. We then
# examined how different choices affect the plan and prediction scores.
#
# There is more to learn about skrub choices than what is covered here.
# In particular, choices are not limited to choosing estimators and
# their hyperparameters: they can be used anywhere DataOps are used,
# such as the argument of a :func:`deferred` function, or the argument of
# other DataOps' methods or operators. Additionally, choices can be
# inter-dependent. Find more information in the :ref:`user guide
# <user_guide_data_ops_tuning_validating_dataops>`.
