"""
Skrub as preprocessing for Table Foundation Models
==================================================

`TabICL <https://github.com/soda-inria/tabicl>`_ is an open source tabular
foundation model: its a powerful predictor that doesn't need hyper-parameter tuning. In this example, skrub takes care of
preparing the dataframe, and TabICL makes the predictions.

.. note::
    This example requires ``tabicl``, which is not installed with skrub.
    You can install it with ``pip install tabicl``.


The TabICL documentation also has a tutorial on how to handle string columns
with skrub: `TabICL string handling tutorial
<https://tabicl.readthedocs.io/en/latest/tutorials/string_handling.html>`_.
"""

# %%
# Loading the data
# ----------------
#
# We use the employee salaries dataset: the goal is to predict the salary of
# an employee from information such as their department, position title and
# hiring date. We only keep the first 500 rows to keep the memory pressure
# and runtime of the example low.
import skrub
from skrub.datasets import fetch_employee_salaries

data = fetch_employee_salaries()
X, y = data.X.iloc[:500], data.y.iloc[:500]
skrub.TableReport(X)

# %%
# Building the pipeline
# ---------------------
#
# :func:`~skrub.tabular_pipeline` recognizes TabICL estimators and configures
# the :class:`~skrub.TableVectorizer` with the following configuration:
#
# .. code-block:: python
#
#     skrub.TableVectorizer(
#         low_cardinality="passthrough",
#         high_cardinality=skrub.StringEncoder(),
#         cardinality_threshold=10,
#         datetime=skrub.DatetimeEncoder(periodic_encoding="spline"),
#     )
#
# String columns with at most 10 unique values are passed through as-is, since
# TabICL handles them natively, while columns with more unique values are
# encoded with a :class:`~skrub.StringEncoder`. TabICL handles missing values
# and scaling by itself, so no imputer or scaler is added.
from tabicl import TabICLRegressor

# To keep this example fast, we use a single estimator. Results would probably
# be a bit better with the default ``n_estimators=8``, but it is more expensive.
model = skrub.tabular_pipeline(TabICLRegressor(n_estimators=1))
model

# %%
# Evaluating the model
# --------------------
#
# We evaluate the pipeline with a 3-fold cross-validation.
from sklearn.model_selection import cross_validate

tabicl_results = cross_validate(model, X, y, cv=3)
scores = tabicl_results["test_score"]
print(f"R2 score: {scores.mean():.3f} ± {scores.std():.3f}")

# %%
# Comparing with the default pipeline
# -----------------------------------
#
# Passing ``"regressor"`` to :func:`~skrub.tabular_pipeline` gives a pipeline
# that ends with a :class:`~sklearn.ensemble.HistGradientBoostingRegressor`.
# The data preparation is still handled by skrub: switching the model is a
# one-line change.
hgb_results = cross_validate(skrub.tabular_pipeline("regressor"), X, y, cv=3)

named_results = [("TabICL", tabicl_results), ("HGBT", hgb_results)]
for name, results in named_results:
    scores = results["test_score"]
    duration = results["fit_time"].mean() + results["score_time"].mean()
    print(
        f"{name:<7} R2 score: {scores.mean():.3f} ± {scores.std():.3f}"
        f"  (fit + predict time: {duration:.1f} s)"
    )

# %%
# We use a boxplot to visualize the distribution of R2 scores across folds
# for each model.
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
names, scores = zip(*[(name, result["test_score"]) for name, result in named_results])
ax.boxplot(scores, orientation="horizontal")
ax.set_yticks(range(1, len(names) + 1), labels=list(names), size=12)
ax.set_xlabel("R2 score", size=14)
ax.set_title("R2 distribution across folds (higher is better)", size=14)
plt.show()

# %%
# On this small subset, TabICL gives better predictions than the default
# pipeline, at the cost of a longer computation time. This comparison is
# only meant to show how easily models can be swapped: results depend a lot
# on the dataset and its size.
#
# Note that TabICL does not need hyperparameter tuning: it works well with its
# default parameters. The data preparation, however, still has parameters that
# can matter, such as the number of components of the
# :class:`~skrub.StringEncoder`. These can be tuned like in any scikit-learn
# pipeline, for example with :class:`~sklearn.model_selection.GridSearchCV`.

# %%
# Conclusion
# ----------
#
# skrub prepares the dataframe for TabICL in a single line with
# :func:`~skrub.tabular_pipeline`, which also makes it easy to compare TabICL
# with other models.
