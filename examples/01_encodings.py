"""
.. _example_encodings:

=====================================================================
Encoding: from a dataframe to a numerical matrix for machine learning
=====================================================================

This example shows how to transform a rich dataframe with columns of various types
into a numerical matrix on which machine-learning algorithms can be applied.
We study the case of predicting wages using the
`employee salaries <https://www.openml.org/d/42125>`_ dataset.

.. |TableVectorizer| replace::
    :class:`~skrub.TableVectorizer`

.. |Pipeline| replace::
    :class:`~sklearn.pipeline.Pipeline`

.. |OneHotEncoder| replace::
     :class:`~sklearn.preprocessing.OneHotEncoder`

.. |GapEncoder| replace::
    :class:`~skrub.GapEncoder`

.. |DatetimeEncoder| replace::
    :class:`~skrub.DatetimeEncoder`

.. |HGBR| replace::
    :class:`~sklearn.ensemble.HistGradientBoostingRegressor`

.. |RandomForestRegressor| replace::
     :class:`~sklearn.ensemble.RandomForestRegressor`

.. |permutation importances| replace::
     :func:`~sklearn.inspection.permutation_importance`
"""

###############################################################################
# Easily encoding a dataframe
# ---------------------------
#
# Let's first retrieve the dataset:
# We denote *X*, the employees characteristics (our inputs aka features), and *y*,
# the annual salary (our target column):


from skrub.datasets import fetch_employee_salaries
import pandas as pd

pd.options.display.max_rows = 5

dataset = fetch_employee_salaries()
employees, salaries = dataset.X, dataset.y
employees

###############################################################################
# We observe diverse columns in the dataset:
#   - binary (``'gender'``),
#   - numerical (``'employee_annual_salary'``),
#   - categorical (``'department'``, ``'department_name'``, ``'assignment_category'``),
#   - datetime (``'date_first_hired'``)
#   - dirty categorical (``'employee_position_title'``, ``'division'``).
#
# Most machine-learning algorithms work with arrays of numbers. Therefore our
# complex, heterogeneous table needs to be processed to extract numerical
# features.
# We can do this easily using skrub's |TableVectorizer|

from skrub import TableVectorizer

vectorizer = TableVectorizer()
features = vectorizer.fit_transform(employees)
features

###############################################################################
# From our 8 columns, the |TableVectorizer| has extracted 143 numerical
# features. Most of them are one-hot encoded representations of the categorical
# features. For example, we can see that 3 columns ``gender_F``, ``gender_M``,
# ``gender_nan`` were created to encode the ``'gender'`` column.

###############################################################################
# Before we explore the parameters of the |TableVectorizer| and the choices it
# made, let us note that by performing apropriate transformations on our
# complex data, it allows us to use our table for machine-learning:

from sklearn.ensemble import HistGradientBoostingRegressor

HistGradientBoostingRegressor().fit(features, salaries)

###############################################################################
# The |TableVectorizer| bridges the gap between tabular data and machine-learning
# pipelines and allows us to apply a machine-learning estimator to our table
# without manual data wrangling and feature extraction.
#

###############################################################################
# Inspecting the TableVectorizer
# ------------------------------
#
# The |TableVectorizer| distinguishes between 4 basic types of columns.
# For each kind, it applies a different transformation, which we can configure.
# The types of columns and the default transformations for each of them are:
#
# - numeric columns: simply casting to float32
# - datetime columns: extracting features such as year, day, hour with the |DatetimeEncoder|
# - low-cardinality categorical columns: one-hot encoding
# - high-cardinality categorical columns: a simple and effective text representation pipeline provided by the |GapEncoder|
#

vectorizer

# We can inspect which transformation was chosen for a each column and retrieve the fitted transformer.
# ``vectorizer.transformers_`` gives us a dictionary which maps column names to the corresponding transformer.

vectorizer.transformers_["date_first_hired"]

# We can also see which features in the vectorizer's output were derived from our input column.

vectorizer.input_to_outputs_["date_first_hired"]

###############################################################################

features[vectorizer.input_to_outputs_["date_first_hired"]]

###############################################################################
# We see that ``"date_first_hired"`` has been recognized and processed as a datetime column.
# But looking closer at our original dataframe, it was encoded as a string.

employees["date_first_hired"]

###############################################################################
# Note the ``dtype: object`` in the output above.
# Before applying the transformers we specify, the |TableVectorizer| performs a
# few preprocessing steps.
#
# For example, the "``to_numeric``" step attempts to parse string columns as
# numbers, the "``clean_null_string``" step replaces values commonly used to
# represent missing values such as ``"N/A"`` with actuall ``null``, etc.
# We can also see the list of steps that were relevant for a given column and
# applied to it

vectorizer.input_to_processing_steps_["date_first_hired"]

###############################################################################

vectorizer.input_to_processing_steps_["year_first_hired"]

###############################################################################


###############################################################################
# A simple Pipeline for tabular data
# ----------------------------------
#
# The |TableVectorizer| outputs data that can be understood by a scikit-learn
# estimator. Therefore we can easily build a 2-step scikit-learn ``Pipeline``
# that we can fit, test or cross-validate and that works well on tabular data.

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

pipeline = make_pipeline(TableVectorizer(), HistGradientBoostingRegressor())
scores = cross_val_score(pipeline, employees, salaries)
print(f"R2 score:  mean: {np.mean(scores):.3f}; std: {np.std(scores):.3f}\n")

###############################################################################
# Feature importances in the statistical model
# --------------------------------------------
#
# In this section, after training a regressor, we will plot the feature importances.
#
# .. topic:: Note:
#
#   To minimize computation time, we use the feature importances computed by the
#   |RandomForestRegressor|, but you should prefer |permutation importances|
#   instead (which are less subject to biases).
#
# First, let's train another scikit-learn regressor, the |RandomForestRegressor|:

from sklearn.ensemble import RandomForestRegressor

vectorizer = TableVectorizer()
regressor = RandomForestRegressor(n_estimators=50, max_depth=20, random_state=0)

pipeline = make_pipeline(vectorizer, regressor)
pipeline.fit(employees, salaries)

###############################################################################
# We are retrieving the feature importances:

avg_importances = regressor.feature_importances_
std_importances = np.std(
    [tree.feature_importances_ for tree in regressor.estimators_], axis=0
)
indices = np.argsort(avg_importances)[::-1]

###############################################################################
# And plotting the results:

import matplotlib.pyplot as plt

top_indices = indices[:20]
labels = vectorizer.get_feature_names_out()[top_indices]

plt.figure(figsize=(12, 9))
plt.barh(
    y=labels,
    width=avg_importances[top_indices],
    yerr=std_importances[top_indices],
    color="b",
)
plt.yticks(fontsize=15)
plt.title("Feature importances")
plt.tight_layout(pad=1)
plt.show()

###############################################################################
# We can see that features such the time elapsed since being hired, having a full-time employment, and the position, seem to be the most informative for prediction.
# However, feature importances must not be over-interpreted -- they capture statistical associations `rather than causal effects <https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation>`_.
# Moreover, the fast feature importance method used here suffers from biases favouring features with larger cardinality, as illustrated in a scikit-learn `example <https://scikit-learn.org/dev/auto_examples/inspection/plot_permutation_importance.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-py>`_.
# In general we should prefer |permutation importances|, but it is a slower method.

###############################################################################
# Conclusion
# ----------
#
# In this example, we motivated the need for a simple machine learning
# pipeline, which we built using the |TableVectorizer| and a
# |HGBR|.
#
# We saw that by default, it works well on a heterogeneous dataset.
#
# To better understand our dataset, and without much effort, we were also able
# to plot the feature importances.
