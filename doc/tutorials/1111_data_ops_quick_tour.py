"""
Quick overview of DataOps
=========================

.. currentmodule:: skrub
"""

# %%
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import skrub

# sphinx_gallery_start_ignore
skrub.set_config(data_ops_open_graph_dropdown=True)
# sphinx_gallery_end_ignore

# %%
# Inputs to our pipeline are declared with :func:`var`

employee_data = skrub.var("employee_data")
employee_data

# %%
# Transformation steps are added by calling methods on the intermediate
# results. An important one is :meth:`DataOp.skb.apply`, which applies a
# scikit-learn estimator.

# %%
salary = skrub.var("salary")
pred = employee_data.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(), y=salary
)
pred

# %%
# Note that the methods are accessed through the special attribute ``.skb``:
# for example ``.skb.apply``. We will explain why shortly.

# %%
# Once we have added all the steps, we create a learner: an object similar to a
# scikit-learn estimator with ``fit`` and ``predict`` methods.

train_dataset = skrub.datasets.fetch_employee_salaries(split="train")

learner = pred.skb.make_learner()
learner.fit({"employee_data": train_dataset.X, "salary": train_dataset.y})

# %%
# Regular scikit-learn estimators always take the same fixed inputs: X and y.
# Skrub learners can process arbitrary data, so the signature of methods like
# ``fit`` and ``predict`` is different: we pass a dictionary of inputs. The
# keys correspond to the names of the variables we used to define our learner.

# %%
test_dataset = skrub.datasets.fetch_employee_salaries(split="test")
learner.predict({"employee_data": test_dataset.X, "salary": test_dataset.y})

# %%
# Cross-validation
# ----------------
#
# We now make a few refinements on the previous pipeline. DataOps can accept
# any type of input and perform all processing, so we will extend our pipeline
# so that it includes the data loading and the creation of our features
# ``employee_data`` and our target ``salary``. The input will be simply the
# path to a csv file:

train_dataset.path

# %%
# Therefore we declare a new variable. We also introduce an important feature
# of DataOps: interactive preview results. If we pass a value to our variable
# when creating it, it is used as example data on which to run our pipeline as
# we define it, so we can see what the result looks like every step of the way.

# %%
csv_path = skrub.var("csv_path", train_dataset.path)
csv_path

# %%
# Similarly to ``.skb.apply`` (which applies an estimator), ``.skb.apply_func``
# applies a function.

# %%
full_data = csv_path.skb.apply_func(pd.read_csv)
full_data

# %%
employee_data = full_data.drop(
    columns="current_annual_salary", errors="ignore"
).skb.mark_as_X()

# %%
salary = full_data["current_annual_salary"].skb.mark_as_y()
pred = employee_data.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(), y=salary
)
pred

# %%
pred.skb.cross_validate(scoring="neg_mean_absolute_percentage_error")

# %%
learner = pred.skb.make_learner(fitted=True)
learner.predict({"csv_path": test_dataset.path})

# %%
# Tuning arbitrary choices
# ------------------------

# %%
full_data = skrub.var("csv_path", train_dataset.path).skb.apply_func(pd.read_csv)
employee_data = full_data.drop(
    columns="current_annual_salary", errors="ignore"
).skb.mark_as_X()
salary = full_data["current_annual_salary"].skb.mark_as_y()

# %%
encoder = skrub.choose_from(
    {"lse": skrub.StringEncoder(), "minhash": skrub.MinHashEncoder()}, name="encoder"
)
pred = employee_data.skb.apply(
    skrub.TableVectorizer(high_cardinality=encoder)
).skb.apply(
    HistGradientBoostingRegressor(
        learning_rate=skrub.choose_float(0.01, 0.7, log=True, name="learning_rate")
    ),
    y=salary,
)
pred.skb.describe_param_grid()

# %%
search = pred.skb.make_randomized_search(backend="optuna", fitted=True)
search.results_

# %%
search.plot_results()

# %%
search.predict({"csv_path": test_dataset.path})
