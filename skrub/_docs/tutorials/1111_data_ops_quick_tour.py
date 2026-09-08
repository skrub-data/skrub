"""
Quick overview of DataOps
=========================

.. currentmodule:: skrub

Here we give a bird's eye view of the DataOps workflow on a simple regression task
that we saw in an :ref:`early example <example_encodings>`: predicting the
salaries of US Government employees.

This dataset is so simple that it can be handled without the DataOps, using a
scikit-learn :class:`~sklearn.pipeline.Pipeline`, but we will move on to more
challenging datasets in later sections.
"""

# %%
# Here is the dataset we will work with. The column to predict is
# ``current_annual_salary``.

# sphinx_gallery_start_ignore
import skrub

skrub.set_config(data_ops_open_graph_dropdown=True, table_report_n_rows=5)
# sphinx_gallery_end_ignore

import skrub

train_dataset = skrub.datasets.fetch_employee_salaries(split="train")
skrub.TableReport(train_dataset.employee_salaries)

# %%
# A first simple pipeline
# -----------------------
#
# We start by defining our predictive pipeline. We will need to encode the
# features with a :class:`TableVectorizer` then predict with a
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor`.
#
# Inputs to our pipeline are declared with :func:`var`:

# %%
employee_data = skrub.var("employee_data")
employee_data

# %%
# Transformation steps are added by calling methods on the intermediate
# results. An important one is :meth:`DataOp.skb.apply`, which applies a
# scikit-learn estimator:

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

salary = skrub.var("salary")

pred = employee_data.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(), y=salary
)
pred

# %%
# Note that the methods are accessed through the special attribute ``.skb``:
# for example ``.skb.apply``. We will explain why shortly.

# %%
# Once we have added all the steps, we create a *learner*: an object similar to a
# scikit-learn estimator with ``fit`` and ``predict`` methods.

learner = pred.skb.make_learner()
learner.fit({"employee_data": train_dataset.X, "salary": train_dataset.y})

# %%
# Regular scikit-learn estimators always take the same fixed inputs: X and y.
# Skrub learners can process arbitrary data, so the signature of methods like
# ``fit`` and ``predict`` is different: we pass a dictionary of inputs. The
# keys correspond to the names of the variables we used to define our learner,
# here ``"employee_data"`` and ``"salary"``.
#
# Finally, we can use our fitted learner to make a prediction:

# %%
test_dataset = skrub.datasets.fetch_employee_salaries(split="test")
learner.predict({"employee_data": test_dataset.X, "salary": test_dataset.y})

# %%
# We can generate a complete report for the execution of our DataOp by calling:
#
# .. code:: python
#
#    pred.skb.full_report({"employee_data": test_dataset.X, "salary": test_dataset.y})
#
# As the output is usually quite large, it does not display inline in a
# notebook but is instead opened in a separate browser tab. However here we
# insert it in the page for convenience. By clicking a node in the graph you
# can see its result, how long it took, and the scikit-learn estimator that was
# fitted (if any).
#
# .. raw:: html
#
#   <iframe
#     id="inlineFrameExample"
#     title="Inline Frame Example"
#     width="130%"
#     height="800"
#     style="transform: scale(0.75); transform-origin: 0 0; border: 2px solid black; margin-bottom: -190px;" # noqa: E501
#     src="../_static/employee_salaries_report/index.html">
#   </iframe>
#
# You can also visit the report
# `here <../_static/employee_salaries_report/index.html>`_.
#
# It is also possible to create report about the execution of specific methods
# of the learner like "fit" and "predict" with :meth:`SkrubLearner.report`.

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
# Therefore we declare a new variable, to represent the CSV path.
#
# We also introduce an important feature of DataOps: interactive preview
# results. If we pass a value to our variable when creating it, it is used as
# example data on which skrub runs our pipeline as we define it, so we can see what
# the result looks like every step of the way.

# %%
csv_path = skrub.var("csv_path", train_dataset.path)
csv_path

# %%
# Note the added "Result" section in the output, which shows what the current
# pipeline's output looks like.
#
# Similarly to ``.skb.apply`` (which applies an estimator), ``.skb.apply_func``
# applies a function:

# %%
import pandas as pd

full_data = csv_path.skb.apply_func(pd.read_csv)
full_data

# %%
# Next, from our full dataframe we extract the predictive features and the
# regression target.
#
# The following snippet of code shows 2 important aspects:
#
# - Any methods or operators we access on our DataOp ``full_data``, like
#   ``drop`` or the ``[]`` operator below, are recorded in the pipeline and
#   will be applied to the DataOp's result:
#   ``full_data['current_annual_salary']`` is roughly equivalent to
#   ``full_data.skb.apply_func(lambda df: df['curent_annual_salary'])``. This
#   is why all the skrub functionality is behind the ``.skb`` prefix as
#   mentioned earlier: all other attribute access will be replayed directly on
#   the result that the DataOp produces.
# - Once we have defined the features and targets, we mark them with
#   :meth:`DataOp.skb.mark_as_X()` and :meth:`DataOp.skb.mark_as_y()`
#   respectively. This tells skrub that when performing cross-validation, those
#   are the intermediate results that should be divided into training and
#   testing sets. Therefore, X and y do not need to be constructed and split
#   *outside* the pipeline. Instead, our pipeline can encompass the full
#   processing, and we indicate where the train/test split should happen.

employee_data = full_data.drop(
    columns="current_annual_salary", errors="ignore"
).skb.mark_as_X()
# (errors='ignore' because this column could be absent at the inference stage.)

salary = full_data["current_annual_salary"].skb.mark_as_y()
salary

# %%
# Finally, we apply the regressor. Note that the X and y nodes, on which
# train/test split is performed, are colored differently.

pred = employee_data.skb.apply(skrub.TableVectorizer()).skb.apply(
    HistGradientBoostingRegressor(), y=salary
)
pred

# %%
# Once we have defined our pipeline, we can tell skrub to perform the
# cross-validation with :meth:`DataOp.skb.cross_validate`.

pred.skb.cross_validate(scoring="neg_mean_absolute_percentage_error")

# %%
# Note that the variables used in this pipeline are different than the previous
# one: we just have ``"csv_path"`` and not ``"employee_data"`` and ``"salary"``
# like before.

learner = pred.skb.make_learner(fitted=True)
learner.predict({"csv_path": test_dataset.path})

# %%
# Tuning arbitrary choices
# ------------------------
#
# The last feature we present in this first tutorial is hyperparameter tuning.
#
# The start of the pipeline is the same as before:

# %%
full_data = skrub.var("csv_path", train_dataset.path).skb.apply_func(pd.read_csv)
employee_data = full_data.drop(
    columns="current_annual_salary", errors="ignore"
).skb.mark_as_X()
salary = full_data["current_annual_salary"].skb.mark_as_y()

# %%
# We use functions like :func:`choose_from` or :func:`choose_float` whenever we
# have a choice for which we want to try several options and keep the one that
# performs best on the validation data.
#
# We simply replace the value by the special "choice" object produced by skrub
# in our pipeline, and it becomes a tunable hyperparameter of our skrub
# learner. Here we want to tune:
#
# - the choice of encoder applied to high-cardinality categorical columns
#   (:class:`StringEncoder` or :class:`~sklearn.preprocessing.TargetEncoder`)
# - for the StringEncoder, the number of components
# - the learning rate of
#   the :class:`~sklearn.ensemble.HistGradientBoostingRegressor`.
#
# **Note:** choices are not restricted to estimators or their hyperparameters,
# we can tune any value used anywhere in a pipeline, or the choice between
# different pipelines; more details :ref:`here <user_guide_data_ops_nesting_choices>`.

from sklearn.preprocessing import TargetEncoder

n_components = skrub.choose_int(10, 80, name="n_components")  # choose int in [10, 80[

encoder = skrub.choose_from(  # choosing between 2 different estimators
    {
        "lse": skrub.StringEncoder(n_components=n_components),  # nesting choices
        "target": TargetEncoder(),
    },
    name="encoder",
)

pred = employee_data.skb.apply(
    skrub.TableVectorizer(high_cardinality=encoder), y=salary
).skb.apply(
    HistGradientBoostingRegressor(
        learning_rate=skrub.choose_float(0.01, 0.7, log=True, name="learning_rate")
    ),
    y=salary,
)
print(pred.skb.describe_param_grid())

# %%
# To actually run the search for the best hyperparameters, we use
# :meth:`DataOp.skb.make_randomized_search` or
# :meth:`DataOp.skb.make_grid_search`. For the randomized search we can use the
# powerful Optuna library which provides features like state-of-the-art
# hyperparameter samplers, live interactive visualization of the search with
# ``optuna-dashboard``, stopping and resuming searches, etc.

search = pred.skb.make_randomized_search(
    backend="optuna", fitted=True, n_iter=16, random_state=0
)
search.results_

# %%
search.plot_results()

# %%
# The search can be used with the same interface as the :class:`SkrubLearner`
# we saw before. Alternatively, we can access its ``best_learner_`` attribute,
# which is a SkrubLearner.

search.predict({"csv_path": test_dataset.path})
