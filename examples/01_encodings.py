"""
.. _example_encodings:

=====================================================================
Encoding: from a dataframe to a numerical matrix for machine learning
=====================================================================

This example demonstrates how to transform a somewhat complicated dataframe
to a matrix well suited for machine-learning. We study the case of predicting wages
using the `employee salaries <https://www.openml.org/d/42125>`_ dataset.

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
# A simple prediction pipeline
# ----------------------------
#
# Let's first retrieve the dataset:

from skrub.datasets import fetch_employee_salaries

dataset = fetch_employee_salaries()

###############################################################################
# We denote *X*, employees characteristics (our input data), and *y*,
# the annual salary (our target column):

X = dataset.X
y = dataset.y

X

###############################################################################
# We observe diverse columns in the dataset:
#   - binary ('gender'),
#   - numerical ('employee_annual_salary'),
#   - categorical ('department', 'department_name', 'assignment_category'),
#   - datetime ('date_first_hired')
#   - dirty categorical ('employee_position_title', 'division').
#
# Using skrub's |TableVectorizer|, we can now already build a machine-learning
# pipeline and train it:

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from skrub import TableVectorizer

pipeline = make_pipeline(TableVectorizer(), HistGradientBoostingRegressor())
pipeline.fit(X, y)

###############################################################################
# What just happened here?
#
# We actually gave our dataframe as an input to the |TableVectorizer| and it
# returned an output useful for the scikit-learn model.
#
# Let's explore the internals of our encoder, the |TableVectorizer|:

from pprint import pprint

# Recover the TableVectorizer from the Pipeline
tv = pipeline.named_steps["tablevectorizer"]

pprint(tv.transformers_)

###############################################################################
# We observe it has automatically assigned an appropriate encoder to
# corresponding columns:

###############################################################################
#     - The |OneHotEncoder| for low cardinality string variables, the columns
#       'gender', 'department', 'department_name' and 'assignment_category'.

tv.named_transformers_["low_card_cat"].get_feature_names_out()

###############################################################################
#     - The |GapEncoder| for high cardinality string columns, 'employee_position_title'
#       and 'division'. The |GapEncoder| is a powerful encoder that can handle dirty
#       categorical columns.

tv.named_transformers_["high_card_cat"].get_feature_names_out()

###############################################################################
#     - The |DatetimeEncoder| to the 'date_first_hired' column. The |DatetimeEncoder|
#       can encode datetime columns for machine learning.

tv.named_transformers_["datetime"].get_feature_names_out()

###############################################################################
# As we can see, it gave us interpretable column names.
#
# In total, we have a reasonable number of encoded columns:

feature_names = tv.get_feature_names_out()

len(feature_names)

###############################################################################
# Let's look at the cross-validated R2 score of our model:

from sklearn.model_selection import cross_val_score
import numpy as np

scores = cross_val_score(pipeline, X, y)
print(f"R2 score:  mean: {np.mean(scores):.3f}; std: {np.std(scores):.3f}\n")

###############################################################################
# The simple pipeline applied on this complex dataset gave us very good results.

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

regressor = RandomForestRegressor()

pipeline = make_pipeline(TableVectorizer(), regressor)
pipeline.fit(X, y)

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
labels = np.array(feature_names)[top_indices]

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
# We can deduce from this data that the three factors that define the most
# the salary are: being hired for a long time, being a manager,
# and having a permanent, full-time job.

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
