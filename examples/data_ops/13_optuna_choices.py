"""
.. currentmodule:: skrub
.. _example_optuna_choices:

Tuning DataOps with Optuna
==========================

This example shows how to use `Optuna
<https://optuna.readthedocs.io/en/stable/>`_ to tune the hyperparameters a
skrub :class:`DataOp`. As seen in the previous example, skrub DataOps can contain
"choices", objects created with :func:`choose_from`, :func:`choose_int`,
:func:`choose_float`, etc. and we can use hyperparameter search techniques to
pick the best outcome for each choice. Performing this search with Optuna
allows us to benefit from its many features such as state-of-the-art search
strategies, monitoring and visualization, stopping and resuming searches, and
parallel or distributed computation.
"""

# %%
# A simple regressor and example data.
# ------------------------------------
#
# We will fit a regressor containing a few choices on a toy dataset. We
# try 2 regressors: gradient boosting and random forest. They both have
# hyperparameters that we want to tune.

# %%
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

import skrub

X, y = skrub.X(), skrub.y()
hgb = HistGradientBoostingRegressor(
    learning_rate=skrub.choose_float(0.01, 0.6, log=True, name="learning_rate")
)
rf = RandomForestRegressor(
    n_estimators=skrub.choose_int(20, 100, name="n_estimators"),
)
regressor = skrub.choose_from({"hgb": hgb, "random forest": rf}, name="regressor")
pred = X.skb.apply(regressor, y=y)
print(pred.skb.describe_param_grid())

# %%
# Load data for the example

# %%
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold

env = {}
env["X"], env["y"] = fetch_california_housing(return_X_y=True, as_frame=True)
cv = KFold(n_splits=4, shuffle=True, random_state=0)

# %%
# Selecting the best hyperparameters with Optuna.
# -----------------------------------------------
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
import optuna


def objective(trial):
    learner = pred.skb.make_learner(choose=trial)
    cv_results = skrub.cross_validate(learner, environment=env, cv=cv)
    return cv_results["test_score"].mean()


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=16)

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
# Exploring the search results
# ----------------------------
#
# Many reporting capabilities are available for example with
# `optuna.visualization
# <https://optuna.readthedocs.io/en/stable/reference/visualization/>`_ or
# `optuna-dashboard
# <https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html>`_.
# The Study object itself has some useful attributes:

# %%
study.best_params

# %%
study.trials_dataframe().sort_values("value", ascending=False).filter(
    regex="(value|params.*)"
)

# %%
# We can see that the histogram gradient boosting seems to perform better than
# the random forest, and that the best learning rate for this dataset seems to
# be inside the range we explored.

# %%
# As a small example of the visalization capabilities we plot the score
# depending on the learning rate of the histogram gradient boosting:

# %%
optuna.visualization.plot_slice(study, params=["0:learning_rate"])
