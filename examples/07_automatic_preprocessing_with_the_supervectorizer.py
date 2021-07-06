"""
Automatic pre-processing with the SuperVectorizer
=================================================

In this notebook, we introduce the `SuperVectorizer`, which automatically
applies transformers to the different columns of a dataset.
We demonstrate that on the `employee salaries` dataset.


.. |OneHotEncoder| replace::
    :class:`~sklearn.preprocessing.OneHotEncoder`

.. |ColumnTransformer| replace::
    :class:`~sklearn.compose.ColumnTransformer`

.. |RandomForestRegressor| replace::
    :class:`~sklearn.ensemble.RandomForestRegressor`

.. |SE| replace:: :class:`~dirty_cat.SimilarityEncoder`

.. |permutation importances| replace::
    :func:`~sklearn.inspection.permutation_importance`

"""

###############################################################################
# Importing the data
# ---------------
# Let's fetch the dataset, and load X and y:

import pandas as pd
from sklearn.model_selection import train_test_split
from dirty_cat.datasets import fetch_employee_salaries

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

employee_salaries = fetch_employee_salaries()
print(employee_salaries['DESCR'])

X: pd.DataFrame = employee_salaries['data']
y = employee_salaries['target']
# We'll drop a few columns we don't want
X.drop(
    [
        'Current Annual Salary',  # Too linked with target
        'full_name',  # Not relevant to the analysis
        '2016_gross_pay_received',  # Too linked with target
        '2016_overtime_pay',  # Too linked with target
        'date_first_hired'  # Redundant with "year_first_hired"
    ],
    axis=1,
    inplace=True
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

###############################################################################
# Using the vectorizer
# --------------------
# Here is a simple workflow with the `SuperVectorizer` using a `Pipeline`:
#
# It's the typical and recommended way of using it.

import numpy as np

# For scikit-learn 0.24, we need to require the experimental feature
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.pipeline import Pipeline

from dirty_cat import SuperVectorizer


pipeline = Pipeline([
    ('vectorizer', SuperVectorizer(auto_cast=True)),
    ('clf', HistGradientBoostingRegressor(random_state=42))
])

###############################################################################
# For reference, let's perform a cross-validation

from sklearn.model_selection import cross_val_score


scores = cross_val_score(pipeline, X, y, scoring='r2')

print(f'{scores=}')
print(f'mean={np.mean(scores)}')
print(f'std={np.std(scores)}')

###############################################################################
# Compared to
# :ref:`example 02<sphx_glr_auto_examples_02_fit_predict_plot_employee_salaries.py>`,
# the mean score is a pretty much the same as the all-|SE|.

###############################################################################
# Analyzing what it does
# ----------------------
# Let's perform the same workflow, but without the `Pipeline`, so we can
# analyze its mechanisms along the way.

from sklearn.ensemble import RandomForestRegressor


sup_vec = SuperVectorizer(auto_cast=True)
regressor = RandomForestRegressor(n_estimators=25, random_state=42)
# Fit the SuperVectorizer
X_train_enc = sup_vec.fit_transform(X_train, y_train)
X_test_enc = sup_vec.transform(X_test)
# And the regressor
regressor.fit(X_train_enc, y_train)

###############################################################################
# Inspecting the features created
# -------------------------------
# Once it has been trained on data,
# we can print the transformers and the columns assignment it creates:

print(sup_vec.transformers_)

###############################################################################
# This is what is being passed to the |ColumnTransformer| under the hood.
# If you're familiar with how the later works, it should be very intuitive.
# We can notice it classified the columns "gender" and "assignment_category"
# as low cardinality string variables.
# A |OneHotEncoder| will be applied to these columns.
#
# The vectorizer actually makes the difference between string variables
# (data type ``object`` and ``string``) and categorical variables
# (data type ``category``).
#
# Next, we can have a look at the encoded feature names.

# Before:
print(X.columns.to_list())
# After :
feature_names = sup_vec.get_feature_names()
print(feature_names[:8])
print(len(feature_names))

###############################################################################
# As we can see, it created a new column for each unique value.
# This is because we used |SimilarityEncoder| on the column "division",
# which was classified as a high cardinality string variable.
# (default values, see `SuperVectorizer`'s docstring).
# In total, we have 1212 encoded columns.
#
# Finally, let's plot the feature importances.
#
# .. topic:: Note:
#
#    we will plot the feature importances computed by the
#    |RandomForestRegressor|, but you should use |permutation importances|
#    instead (which are much more accurate)
#
#    We chose the former over the later for the sake of performance.

import matplotlib.pyplot as plt


# Getting feature importances
importances = regressor.feature_importances_
std = np.std(
    [
        tree.feature_importances_
        for tree in regressor.estimators_
    ],
    axis=0
)
indices = np.argsort(importances)[::-1]

# Plotting the results:

plt.figure(figsize=(18, 9))
plt.title("Feature importances")
n = 20
n_indices = indices[:n]
labels = np.array(feature_names)[n_indices]
plt.barh(range(n), importances[n_indices], color="b", yerr=std[n_indices])
plt.yticks(range(n), labels, size=15)
plt.tight_layout(pad=1)
plt.show()

###############################################################################
# We can deduce a few things from this data:
# the three factors that define the most the salary are: being a manager,
# being hired for a long time, and have a permanent, full-time job.
