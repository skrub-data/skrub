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
# Let's first retrieve the dataset, using the one of the downloaders from the :mod:`skrub.datasets` module.
# As all the downloaders, :func:`~skrub.datasets.fetch_employee_salaries` returns a dataset with attributes ``X``, and ``y``.
# ``X`` is a dataframe which contains the features (aka design matrix, explanatory variables, independent variables).
# ``y`` is a column (pandas Series) which contains the target (aka dependent, response variable) that we want to learn to predict from ``X``.
# In this case ``y`` is the annual salary.

from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()
employees, salaries = dataset.X, dataset.y
employees

###############################################################################
salaries

###############################################################################
# We observe diverse columns in the ``employees`` dataframe:
#   - numeric (``'year_first_hired'``)
#   - dates (``'date_first_hired'``)
#   - low-cardinality categorical (``'gender'``, ``'department'``, ``'department_name'``, ``'assignment_category'``)
#   - high-cardinality categorical (``'employee_position_title'``, ``'division'``).
#
# Most machine-learning algorithms work with arrays of numbers.
# Therefore our complex, heterogeneous table needs to be processed to extract numeric features.
# Transforming a complex real-world object such as a date into a vector of numeric features —more adequate for machine learning— is often called *vectorizing* it.
#
# We can do this easily using skrub's |TableVectorizer|.

from skrub import TableVectorizer

vectorizer = TableVectorizer()
vectorized_employees = vectorizer.fit_transform(employees)
vectorized_employees

###############################################################################
# From our 8 columns, the |TableVectorizer| has extracted 143 numerical
# features. Most of them are one-hot encoded representations of the categorical
# features. For example, we can see that 3 columns ``gender_F``, ``gender_M``,
# ``gender_nan`` were created to encode the ``'gender'`` column.

###############################################################################
# By performing apropriate transformations on our complex data, the |TableVectorizer| produced numeric features that we can use for machine-learning:

from sklearn.ensemble import HistGradientBoostingRegressor

HistGradientBoostingRegressor().fit(vectorized_employees, salaries)

###############################################################################
# The |TableVectorizer| bridges the gap between tabular data and machine-learning pipelines.
# It allows us to apply a machine-learning estimator to our dataframe without manual data wrangling and feature extraction.
#

###############################################################################
# Inspecting the TableVectorizer
# ------------------------------
#
# The |TableVectorizer| distinguishes between 4 basic kinds of columns (more may be added in the future).
# For each kind, it applies a different transformation, which we can configure.
# The kinds of columns and the default transformation for each of them are:
#
# - numeric columns: simply casting to floating-point
# - datetime columns: extracting features such as year, day, hour with the |DatetimeEncoder|
# - low-cardinality categorical columns: one-hot encoding
# - high-cardinality categorical columns: a simple and effective text representation pipeline provided by the |GapEncoder|

vectorizer

###############################################################################
# We can inspect which transformation was chosen for a each column and retrieve the fitted transformer.
# ``vectorizer.column_kinds_`` provides an overview of how the vectorizer categorized columns in our input:

vectorizer.column_kinds_

###############################################################################
# ``vectorizer.transformers_`` gives us a dictionary which maps column names to the corresponding transformer.

vectorizer.transformers_["date_first_hired"]

###############################################################################
# We can also see which features in the vectorizer's output were derived from a given input column.

vectorizer.input_to_outputs_["date_first_hired"]

###############################################################################

vectorized_employees[vectorizer.input_to_outputs_["date_first_hired"]]

###############################################################################
# Finally, we can go in the opposite direction: given a column in the input, find out from which input column it was derived.

vectorizer.output_to_input_["department_BOA"]


###############################################################################
# Dataframe preprocessing
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# Note that ``"date_first_hired"`` has been recognized and processed as a datetime column.

vectorizer.column_kinds_["datetime"]

###############################################################################
# But looking closer at our original dataframe, it was encoded as a string.

employees["date_first_hired"]

###############################################################################
# Note the ``dtype: object`` in the output above.
# Before applying the transformers we specify, the |TableVectorizer| performs a few preprocessing steps.
#
# For example, strings commonly used to represent missing values such as ``"N/A"`` are replaced with actual ``null``.
# As we saw above, columns containing strings that represent dates (e.g. ``'2024-05-15'``) are detected and converted  to proper datetimes.
#
# We can inspect the list of steps that were applied to a given column:

vectorizer.all_processing_steps_["date_first_hired"]

###############################################################################

vectorizer.all_processing_steps_["year_first_hired"]

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
