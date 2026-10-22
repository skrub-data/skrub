"""
Using TabICL with skrub
=======================

`TabICL <https://github.com/soda-inria/tabicl>`_ is an open source tabular
foundation model: it makes predictions on a new table without any
dataset-specific training of its weights. In this example, skrub takes care of
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
# hiring date. We only keep the first 1000 rows.
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
# the :class:`~skrub.TableVectorizer` accordingly: low-cardinality string
# columns are passed through as-is, high-cardinality string columns are encoded
# with a :class:`~skrub.StringEncoder`, and datetimes are encoded with a
# :class:`~skrub.DatetimeEncoder`. TabICL handles missing values and scaling
# by itself, so no imputer or scaler is added.
from tabicl import TabICLRegressor

model = skrub.tabular_pipeline(TabICLRegressor())
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

for name, results in [("TabICL", tabicl_results), ("HGBT", hgb_results)]:
    scores = results["test_score"]
    duration = results["fit_time"].mean() + results["score_time"].mean()
    print(
        f"{name:<7} R2 score: {scores.mean():.3f} ± {scores.std():.3f}"
        f"  (fit + predict time: {duration:.1f} s)"
    )

# %%
# On this small subset, TabICL gives better predictions than the default
# pipeline, at the cost of a longer computation time. This comparison is
# only meant to show how easily models can be swapped: results depend a lot
# on the dataset and its size.

# %%
# Conclusion
# ----------
#
# skrub prepares the dataframe for TabICL in a single line with
# :func:`~skrub.tabular_pipeline`, which also makes it easy to compare TabICL
# with other models.
