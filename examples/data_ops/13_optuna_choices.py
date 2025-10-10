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
# Defining the pipeline and creating example data
# -----------------------------------------------
#
# We will fit a binary classifier containing a few choices on a toy dataset. We
# try several classifiers: logisitic regression, random forest, and a dummy
# baseline. The logistic regression and random forest have hyperparameters that
# we want to tune.

# %%
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import skrub

X, y = skrub.X(), skrub.y()
logistic = LogisticRegression(C=skrub.choose_float(0.1, 10.0, log=True, name="C"))
rf = RandomForestClassifier(
    n_estimators=skrub.choose_int(3, 30, name="n estimators"), random_state=0
)
classifier = skrub.choose_from(
    {"logistic": logistic, "random forest": rf, "dummy": DummyClassifier()},
    name="classifier",
)
pred = X.skb.apply(classifier, y=y)
print(pred.skb.describe_param_grid())

# %%
# Create some toy data for the example

# %%
from sklearn.datasets import make_classification

X_a, y_a = make_classification(random_state=0)
env = {"X": X_a, "y": y_a}

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
# Here we return a single score (accuracy), but multi-objective
# optimization is also possible. Please refer to the Optuna documentation for
# more information.

# %%
import optuna

study = optuna.create_study(direction="maximize")


def objective(trial):
    # The optuna trial picks an outcome for each choice in the DataOp
    learner = pred.skb.make_learner(choose=trial)
    # We score the resulting SkrubLearner
    cv_results = skrub.cross_validate(learner, environment=env)
    # We return the score to the Study
    return cv_results["test_score"].mean()


study.optimize(objective, n_trials=30)
study.trials_dataframe().sort_values("value", ascending=False).filter(
    regex="(value|params.*)"
)

# %%
# We can see that the random forest performs much better than the logistic
# regression, which in turns is much better than the dummy classifier, and that
# ``n_estimators=10`` works well for the random forest.
#
# Finally we build a learner with the best hyperparameters identified by Optuna
# (stored in the ``best_params`` attribute), and fit it on the full dataset. We
# now have a learner fitted with tuned hyperparameters, ready to make
# predictions on unseen data.

# %%
best_learner = pred.skb.make_learner(choose=study.best_trial)

# This would achieve the same result:
# best_learner = pred.skb.make_learner()
# best_learner.set_params(**study.best_params)

best_learner.fit(env)
print(best_learner.describe_params())

# %%
# Many reporting capabilities are available for example with `optuna-dashboard
# <https://optuna-dashboard.readthedocs.io/en/latest/getting-started.html>`_.
# As a small example we plot the score depending on the number of trees in the
# random forest:

# %%
optuna.visualization.plot_slice(study, params=["1:n estimators"])
