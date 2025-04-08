"""
Tuning pipelines
================

Our machine-learning pipeline typically contains some values or choices which
may influence its prediction performance, such as hyperparameters (e.g. the
regularization parameter ``alpha`` of a ``RidgeClassifier``, the
``learning_rate`` of a ``HistGradientBoostingClassifier``), which estimator to
use (e.g. ``RidgeClassifier`` or ``HistGradientBoostingClassifier``), or which
steps to include (e.g. should we join a table to bring additional information or
not).

We want to tune those choices by trying several options and keeping those that
give the best performance on a validation set.

Skrub `expressions <10_expressions.html>`_ provide a convenient way to specify
the range of possible values, by inserting it directly in place of the actual
value. For example we can write:

 ``RidgeClassifier(alpha=skrub.choose_from([0.1, 1.0, 10.0], name='α'))``

instead of:

``RidgeClassifier(alpha=1.0)``.

Skrub then inspects
our pipeline to discover all the places where we used objects like
``skrub.choose_from()`` and builds a grid of hyperparameters for us.

"""

# %%
# We will illustrate hyperparameter tuning on the "toxicity" dataset. This
#  dataset contains 1,000 texts and the task is to predict if they are
#  flagged as being toxic or not.
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
# We mark the ``texts`` column as the input and the ``labels`` column as the target.
# See `the previous example <10_expressions.html>`_ for a more detailed explanation
# of ``skrub.X`` and ``skrub.y``.
# We then encode the text with a ``MinHashEncoder`` and fit a
# ``HistGradientBoostingClassifier`` on the resulting features.

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
# For the sake of the example, we will focus on the number of ``MinHashEncoder``
# components and the ``learning_rate`` of the ``HistGradientBoostingClassifier``
# to illustrate the ``skrub.choose_from(...)`` objects.
# When we use a scikit-learn hyperparameter-tuner like ``GridSearchCV`` or
# ``RandomizedSearchCV``, we need to specify a grid of hyperparameters separately
# from the estimator, with something similar to
# ``GridSearchCV(my_pipeline, param_grid={"encoder__n_components: [5, 10, 20]"})``.
# Instead, with skrub we can use
# ``skrub.choose_from(...)`` directly where the actual value
# would normally go. Skrub then takes care of constructing the
# ``GridSearchCV``'s parameter grid for us.
#
# Several utilities are available:
#
# - ``choose_from`` to choose from a discrete set of values
# - ``choose_float`` and ``choose_int`` to sample numbers in a given range
# - ``choose_bool`` to choose between ``True`` and ``False``
# - ``optional`` to choose between something and ``None``; typically to make a
#   transformation step optional such as
#   ``X.skb.apply(skrub.optional(StandardScaler()))``
#
# Choices can be given a name which is used to display hyperparameter search
# results and plots or to override their outcome.
#
# Note that ``skrub.choose_float()`` and ``skrub.choose_int()`` can be given a
# ``log`` argument to sample in log scale.
# %%
X, y = skrub.X(texts), skrub.y(labels)

encoder = skrub.MinHashEncoder(
    n_components=skrub.choose_int(5, 50, log=True, name="N components")
)
X = X.skb.apply(encoder)

classifier = HistGradientBoostingClassifier(
    learning_rate=skrub.choose_float(0.01, 0.9, log=True, name="lr")
)
pred = X.skb.apply(classifier, y=y)

# %%
# We can then obtain an estimator that performs the hyperparameter search with
# ``.skb.get_grid_search()`` or ``.skb.get_randomized_search()``. They accept
# the same arguments as their scikit-learn counterparts (e.g. ``scoring`` and
# ``n_jobs``). Also, like ``.skb.get_estimator()``, they accept a ``fitted``
# argument and if it is ``True`` the search is fitted on the data we provided
# when initializing our pipeline's variables.

search = pred.skb.get_randomized_search(n_iter=8, n_jobs=4, random_state=1, fitted=True)
search.results_

# %%
# If the plotly library is installed, we can visualize the results of the
# hyperparameter search with ``.plot_results()``.
# In the plot below, each line represents a combination of hyperparameters (in
# this case, only ``N components`` and ``learning rate``), and each column of
# points represents either a hyperparameter, or the score of a given
# combination of hyperparameters.
# The color of the line represents the score of the combination of hyperparameters.
# The plot is interactive, and it is  possible to select only a subset of the
# hyperparameters to visualize by dragging the mouse over each column to select
# the desired range.
# This is particularly useful when there are many combinations of hyperparameters,
# and we are interested in understanding which hyperparameters have the largest
# impact on the score.

# %%
search.plot_results()

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

X = X.assign(
    length=skrub.choose_from(
        {"words": X["text"].str.count(r"\b\w+\b"), "chars": X["text"].str.len()},
        name="length",
    )
)
X

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
n_components = skrub.choose_int(5, 50, log=True, name="N components")

encoder = skrub.choose_from(
    {
        "minhash": skrub.MinHashEncoder(n_components=n_components),
        "lse": skrub.StringEncoder(n_components=n_components),
    },
    name="encoder",
)
X = X.skb.apply(encoder, cols="text")

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
pred = X.skb.apply(classifier, y=y)
print(pred.skb.describe_param_grid())

# %%
search = pred.skb.get_randomized_search(
    n_iter=16, n_jobs=4, random_state=1, fitted=True
)
search.plot_results()

# %%
# Now that we have a more complex pipeline, we can draw more conclusions from the
# parallel coordinate plot. For example, we can see that the
# ``HistGradientBoostingClassifier``
# performs better than the ``RidgeClassifier`` in most cases, that the ``StringEncoder``
# outperforms the ``MinHashEncoder``, and that the choice of the additional ``length``
# feature does not have a significant impact on the score.

# %%
# Advanced usage
# --------------
#
# This section shows some more advanced or less frequently needed use cases.
#
# Choices can depend on each other
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Sometimes not all combinations (cross-product) of hyperparameter values make
# sense, and instead choices may be linked. For example, our downstream estimator
# can be a ``RidgeClassifier`` or ``HistGradientBoostingClassifier``, and
# standard scaling should be applied only when it is a ``Ridge``.
#
# Skrub choices have a ``match`` method to obtain different results depending
# on the outcome of the choice.

# %%
from sklearn.preprocessing import StandardScaler

X, y = skrub.X(texts), skrub.y(labels)

X = X.skb.apply(skrub.MinHashEncoder())

estimator_kind = skrub.choose_from(["ridge", "HGB"], name="estimator kind")

scaling = estimator_kind.match({"ridge": StandardScaler(), "HGB": "passthrough"})
X = X.skb.apply(scaling)

classifier = estimator_kind.match(
    {"ridge": RidgeClassifier(), "HGB": HistGradientBoostingClassifier()}
)
pred = X.skb.apply(classifier, y=y)
print(pred.skb.describe_param_grid())

# %%
# Here we can see that there is only one parameter: the estimator kind. When it
# is ``"ridge"``, the ``StandardScaler`` and the ``RidgeClassifier`` are used;
# when it is ``"HGB"`` ``"passthrough"`` and the
# ``HistGradientBoostingClassifier`` are used.
#
# Similarly, objects returned by ``choose_bool`` have a ``if_else()`` method.

# %%
# Choices can be turned into expressions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can turn a choice (or the result of a choice ``match()`` or ``if_else``)
# into an expression, so that we can keep chaining more operations onto it.
# Here, we create a ``.choose_bool()`` object to choose whether to add the length
# of the text as a feature or not. Then, ``if_else()`` will assign the length
# of the text to a new column ``length`` if the choice is ``True``, or do nothing
# if the choice is ``False``.
# %%
X, y = skrub.X(texts), skrub.y(labels)

add_length = skrub.choose_bool(name="add_length")
X = add_length.if_else(X.assign(length=X["text"].str.len()), X).as_expr()
X = X.skb.apply(skrub.MinHashEncoder(n_components=2), cols="text")

# Note: we can manually set the outcome of a choice when evaluating an
# expression (or fitting an estimator)

X.skb.eval({"add_length": False})

# %%
X.skb.eval({"add_length": True})

# %%
# Arbitrary logic depending on a choice
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# When ``match`` or ``if_else`` are not enough and we need to apply arbitrary,
# eager logic based on a choice we can resort to using ``skrub.deferred``. For
# example the choice of adding the text length or not could also have been
# written as:
# %%
X, y = skrub.X(texts), skrub.y(labels)


@skrub.deferred
def extract_features(df, add_length):
    if add_length:
        return df.assign(length=df["text"].str.len())
    return df


X = extract_features(X, skrub.choose_bool(name="add_length"))
X = X.skb.apply(skrub.MinHashEncoder(n_components=2), cols="text")

X.skb.eval({"add_length": False})

# %%
X.skb.eval({"add_length": True})

# %%
# Concluding, we have seen how to use skrub's ``choose_from`` objects to tune
# hyperparameters, choose optional configurations, add features, and nest choices.
# We then looked at how the different choices affect the pipeline and the prediction
# scores.
#
# Thanks to the ``choose_from`` objects, Skrub expressions ease the process of
# hyperparameter tuning.
