"""
Automatic pre-processing with the SuperVectorizer
=================================================

In this notebook, we introduce the |SV|, which automatically
turns a heterogeneous dataset into a numerical representation, finding
the right transformers to apply to the different columns.

We demonstrate it on the `employee salaries` dataset.

.. |SV| replace::
    :class:`~dirty_cat.SuperVectorizer`

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
# ------------------
# Let's fetch the dataset, and load X (the employees' features) and y
# (the salary to predict):
from dirty_cat.gap_encoder import GapEncoder
from dirty_cat.datasets import fetch_employee_salaries
employee_salaries = fetch_employee_salaries()
print(employee_salaries['DESCR'])

###############################################################################

X = employee_salaries['data']
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

###############################################################################
# The data are in a fairly complex and heterogeneous dataframe:
X

###############################################################################
# The challenge is to turn this dataframe into a form suited for
# machine learning.

###############################################################################
# Using the SuperVectorizer in a supervised-learning pipeline
# ------------------------------------------------------------
#
# Assembling the |SV| in a pipeline with a powerful learner,
# such as gradient boosted trees, gives **a machine-learning method that
# can be readily applied to the dataframe**.
#
# It's the typical and recommended way of using it.


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
# Let's perform a cross-validation to see how well this model predicts

from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, X, y, scoring='r2')

import numpy as np
print(f'{scores=}')
print(f'mean={np.mean(scores)}')
print(f'std={np.std(scores)}')

###############################################################################
# The prediction perform here is pretty much as good as in :ref:`example
# 02<sphx_glr_auto_examples_02_fit_predict_plot_employee_salaries.py>`,
# but the code here is much simpler as it does not involve specifying
# columns manually.

###############################################################################
# Analyzing the features created
# -------------------------------
#
# Let us perform the same workflow, but without the `Pipeline`, so we can
# analyze its mechanisms along the way.
sup_vec = SuperVectorizer(
    auto_cast=True,
    high_card_str_transformer=GapEncoder(n_components=50),
    high_card_cat_transformer=GapEncoder(n_components=50)
)

##############################################################################
# We split the data between train and test, and transform them:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42
)

X_train_enc = sup_vec.fit_transform(X_train, y_train)
X_test_enc = sup_vec.transform(X_test)

###############################################################################
# Inspecting the features created
# .................................
# Once it has been trained on data,
# we can print the transformers and the columns assignment it creates:

sup_vec.transformers_

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
#
# Before encoding:
X.columns.to_list()

###############################################################################
# After encoding (we only plot the first 8 feature names):
feature_names = sup_vec.get_feature_names()
feature_names[:8]

###############################################################################
# As we can see, it created a new column for each unique value.
# This is because we used |SE| on the column "division",
# which was classified as a high cardinality string variable.
# (default values, see |SV|'s docstring).
#
# In total, we have 56 encoded columns.
len(feature_names)


###############################################################################
# Feature importance in the statistical model
# ............................................
# In this section, we will train a regressor, and plot the feature importances
# .. topic:: Note:
#
#    To minimize compute time, use the feature importances computed by the
#    |RandomForestRegressor|, but you should prefer |permutation importances|
#    instead (which are less subject to biases)
#
# First, let's train the |RandomForestRegressor|,

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X_train_enc, y_train)


###############################################################################
# Getting the feature importances
importances = regressor.feature_importances_
std = np.std(
    [
        tree.feature_importances_
        for tree in regressor.estimators_
    ],
    axis=0
)
indices = np.argsort(importances)[::-1]

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
# We can deduce from this data that the three factors that define the
# most the salary are: being a manager, being hired for a long time, and
# have a permanent, full-time job :).
