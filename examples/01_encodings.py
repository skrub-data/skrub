"""
.. _example_encodings:

==========================================================================
Encoding: turning any dataframe to a numerical matrix for machine learning
==========================================================================

This example demonstrates how to transform a somewhat complicated dataframe
to a matrix well suited for machine-learning. We study the case of predicting wages
using the `employee salaries <https://www.openml.org/d/42125>`_ dataset.

Let's dive right in!


.. |TableVectorizer| replace::
    :class:`~skrub.TableVectorizer`

.. |Pipeline| replace::
    :class:`~sklearn.pipeline.Pipeline`

.. |make_pipeline| replace::
    :func:`~sklearn.pipeline.make_pipeline`

.. |cross_validate| replace::
    :func:`~sklearn.model_selection.cross_validate`

.. |OneHotEncoder| replace::
    :class:`~sklearn.preprocessing.OneHotEncoder`

.. |ColumnTransformer| replace::
    :class:`~sklearn.compose.ColumnTransformer`

.. |GapEncoder| replace::
    :class:`~skrub.GapEncoder`

.. |MinHashEncoder| replace::
    :class:`~skrub.MinHashEncoder`

.. |DatetimeEncoder| replace::
    :class:`~skrub.DatetimeEncoder`

.. |HGBR| replace::
    :class:`~sklearn.ensemble.HistGradientBoostingRegressor`

.. |SimilarityEncoder| replace::
    :class:`~skrub.SimilarityEncoder`
"""

###############################################################################
# A simple prediction pipeline
# ----------------------------
#
# Let's first retrieve the dataset:

from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()

dataset.description

###############################################################################
# Alias *X*, the descriptions of employees (our input data), and *y*,
# the annual salary (our target column):

X = dataset.X
y = dataset.y

X

###############################################################################
# We observe a few things from the dataset:
# - We have diverse columns: binary ('gender'), numerical
#   ('employee_annual_salary'), categorical ('department', 'department_name',
#   'assignment_category'), datetime ('date_first_hired') and dirty categories
#   ('employee_position_title', 'division').
#
# Now, without much more investigation, we can already build a machine-learning
# pipeline and train it:

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer

pipeline = make_pipeline(TableVectorizer(), HistGradientBoostingRegressor())
pipeline.fit(X, y)

###############################################################################
# What just happened there?
# -------------------------
#
# First, it did not raise any errors, yay!
# Let's explore the internals of our encoder, the |TableVectorizer|:

from pprint import pprint

# Recover the TableVectorizer from the pipeline
tv = pipeline.named_steps["tablevectorizer"]

pprint(tv.transformers_)

###############################################################################
# We observe it has automatically assigned an appropriate encoder to
# corresponding columns.
# For example, it classified the columns 'gender', 'department',
# 'department_name' and 'assignment_category' as low cardinality
# string variables.
# Two remarkable things:, it has affected a |GapEncoder| to the columns
# `employee_position_title` and `division`, and a |DatetimeEncoder| to the
# 'date_first_hired' column.
#
# The |GapEncoder| is a powerful encoder that can handle dirty
# categorical columns.
# The |DatetimeEncoder| can encode datetime columns for machine learning.
#
# Next, we can have a look at the encoded feature names.
#
# Before encoding:

X.columns.to_list()

##############################################################################
# After encoding (we only plot the first 8 feature names):

feature_names = tv.get_feature_names_out()

feature_names[:8]

###############################################################################
# As we can see, it gave us interpretable columns.
# This is because we used the |GapEncoder| on the column ‘division’,
# which was classified as a high cardinality string variable
# (default values, see |TableVectorizer|’s docstring).
#
# In total, we have a reasonable number of encoded columns:

len(feature_names)


###############################################################################
# Feature importances in the statistical model
# --------------------------------------------
#
# In this section, we will train a regressor, and plot the feature importances.
#
# .. topic:: Note:
#
#   To minimize computation time, we use the feature importances computed by the
#   |RandomForestRegressor|, but you should prefer |permutation importances|
#   instead (which are less subject to biases).
#
# First, let's train the |RandomForestRegressor|:

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor()
regressor.fit(X, y)

###############################################################################
# Retrieving the feature importances:

import numpy as np

importances = regressor.feature_importances_
std = np.std([tree.feature_importances_ for tree in regressor.estimators_], axis=0)
indices = np.argsort(importances)
# Sort from least to most
indices = list(reversed(indices))

###############################################################################
# Plotting the results:

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 9))
plt.title("Feature importances")
n = 20
n_indices = indices[:n]
labels = np.array(feature_names)[n_indices]
plt.barh(range(n), importances[n_indices], color="b", yerr=std[n_indices])
plt.yticks(range(n), labels, size=15)
plt.tight_layout(pad=1)
plt.show()

###############################################################################
# Conclusion
# ----------
#
# In this example, we motivated the need for a simple machine learning
# pipeline, which we built using the |TableVectorizer| and a
# |HistGradientBoostingRegressor|.
#
# We saw that by default, it works well on a heterogeneous dataset.
#
# To better understand our dataset, and without much effort, we were also able
# to plot the feature importances.
