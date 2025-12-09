"""
.. currentmodule:: skrub
.. _example_optuna_choices:

Tuning DataOps with Optuna
==========================

This example shows how to use `Optuna
<https://optuna.readthedocs.io/en/stable/>`_ to tune the hyperparameters of a
skrub :class:`DataOp`. As seen in the previous example, skrub DataOps can contain
"choices", objects created with :func:`choose_from`, :func:`choose_int`,
:func:`choose_float`, etc. and we can use hyperparameter search techniques to
pick the best outcome for each choice. Performing this search with Optuna
allows us to benefit from its many features, such as state-of-the-art search
strategies, monitoring and visualization, stopping and resuming searches, and
parallel or distributed computation.

In order to use Optuna with skrub, the package must be installed first.
This can be done with pip:

.. code-block:: bash

    pip install optuna

"""

# %%
# A simple regressor and example data.
# ------------------------------------
#
# We will fit a regressor containing a few choices on a toy dataset. We
# try 2 regressors: gradient boosting and random forest. They both have
# hyperparameters that we want to tune.

# %%
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

import skrub

extra_tree = ExtraTreesRegressor(
    min_samples_leaf=skrub.choose_int(1, 32, log=True, name="min_samples_leaf"),
)
ridge = Ridge(alpha=skrub.choose_float(0.01, 10.0, log=True, name="α"))

regressor = skrub.choose_from(
    {"extra_tree": extra_tree, "ridge": ridge}, name="regressor"
)
data = skrub.var("data")
X = data.drop(columns="MedHouseVal", errors="ignore").skb.mark_as_X()
y = data["MedHouseVal"].skb.mark_as_y()
pred = X.skb.apply(regressor, y=y)
print(pred.skb.describe_param_grid())

# %%
# Load data for the example

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold

# (We subsample the dataset by half to make the example run faster)
df = fetch_california_housing(as_frame=True).frame.sample(10_000, random_state=0)

# The environment we will use to fit the learners created by our DataOp.
env = {"data": df}
cv = KFold(n_splits=4, shuffle=True, random_state=0)

# %%
# Selecting the best hyperparameters with Optuna.
# -----------------------------------------------
#
# The simplest way to use Optuna is to pass ``backend='optuna'`` to
# :meth:`DataOp.skb.make_randomized_search()`. It is used very similarly as
# with the default backend
# (:class:`sklearn.model_selection.RandomizedSearchCV`). Additional
# parameters are available to control the Optuna sampler, storage and study
# name, and timeout.
# Note that in order to persist the study and resume it later, the ``storage``
# parameter must be set to a valid database URL (e.g., a SQLite file). Refer to
# the User Guide for an example.

# %%
search = pred.skb.make_randomized_search(backend="optuna", cv=cv, n_iter=10)
search.fit(env)
search.results_

# %%
# The usual ``results_``, ``detailed_results_`` and ``plot_results()`` are
# still available.

# %%
search.plot_results()

# %%
# The Optuna :class:`Study <optuna.study.Study>` that was used to run the
# hyperparameter search is available in the attribute ``study_``:

# %%
search.study_

# %%
search.study_.best_params

# %%
# This allows us to use Optuna's reporting capabilities provided in
# `optuna.visualization
# <https://optuna.readthedocs.io/en/stable/reference/visualization/>`_ or
# `optuna-dashboard
# <https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html>`_.

# %%
import optuna

optuna.visualization.plot_slice(search.study_, params=["0:min_samples_leaf"])

# %%
# Using Optuna directly for more advanced use cases
# -------------------------------------------------
#
# Often we may want more control over the use of Optuna, or to access
# functionality not available through :meth:`DataOp.skb.make_randomized_search`
# such as the ask-and-tell interface, trial pruning, callbacks,
# multi-objective optimization, etc. .
#
# Directly using Optuna ourselves is also easy, as we will show now. What makes
# this possible is that we can pass an Optuna Trial to
# :meth:`DataOp.skb.make_learner` in which case the parameters suggested by the
# trial are used to create the learner.
#
# We revisit the example above, following the typical Optuna workflow.
#
# The :class:`optuna.Study <optuna.study.Study>` runs the hyperparameter
# search.
#
# Its method :meth:`optimize <optuna.study.Study.optimize>` is given an
# ``objective`` function. The ``objective`` must accept a
# :class:`~optuna.trial.Trial` object (which is produced by the study and picks
# the parameters for a given evaluation of the objective) and return the value
# to maximize (or minimize).
#
# To use Optuna with a :class:`DataOp`, we just need to pass the Trial object
# to :meth:`DataOp.skb.make_learner`. This creates a :class:`SkrubLearner`
# initialized with the parameters picked by the optuna Trial.
#
# We can then cross-validate the SkrubLearner, or score it however we prefer,
# and return the score so that the optuna Study can take it into account.
#
# Here we return a single score (R²), but multi-objective
# optimization is also possible. Please refer to the Optuna documentation for
# more information.


# %%


def objective(trial):
    learner = pred.skb.make_learner(choose=trial)
    cv_results = skrub.cross_validate(learner, environment=env, cv=cv)
    return cv_results["test_score"].mean()


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
study.best_params

# %%
# We can also use Optuna's visualization capabilities to inspect the study:
optuna.visualization.plot_optimization_history(study)

# %%
# Now we build a learner with the best hyperparameters and fit it on the full
# dataset:

# %%
best_learner = pred.skb.make_learner(choose=study.best_trial)

# This would achieve the same result:
# best_learner = pred.skb.make_learner()
# best_learner.set_params(**study.best_params)

best_learner.fit(env)
print(best_learner.describe_params())

# %%
