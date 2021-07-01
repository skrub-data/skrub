"""
Automatic pre-processing with the SuperVectorizer
=================================================

In this notebook, we will illustrate the use of the `SuperVectorizer`, which
can automatically classify columns of a dataset based on their type, and apply
transformers based on this classification.
To demonstrate that, we will use the `employee salaries` dataset.
"""

###############################################################################
# Import the data
# ---------------
#
# Let's fetch the dataset, and load X and y:
import pandas as pd
from sklearn.model_selection import train_test_split
from dirty_cat.datasets import fetch_employee_salaries

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)


###############################################################################
# Using the SuperVectorizer
# -------------------------
# Here is a simple workflow with the SuperVectorizer.

from sklearn.ensemble import RandomForestRegressor
from dirty_cat import SuperVectorizer

sup_vec = SuperVectorizer(auto_cast=True)
regressor = RandomForestRegressor(n_estimators=25, random_state=42)
# Fit the SuperVectorizer
X_train_enc = sup_vec.fit_transform(X_train, y_train)
X_test_enc = sup_vec.transform(X_test)
# And the regressor
regressor.fit(X_train_enc, y_train)


###############################################################################
# Under the hood
# --------------
# Let's now break down what the SuperVectorizer did.
#
# Once it has been trained on data,
# we can print the assignation it did:

print(sup_vec.transformers_)

# This is what is being passed to the ColumnTransformer under the hood.
# If you're familiar with how the later works, it should be very intuitive.
# We can notice it considered the columns "gender" and "assignment_category"
# as low cardinality string variables.
# The vectorizer actually makes the difference between string variables
# (data type "object") and categorical variables (data type "category").
# A OneHotEncoder() will be applied to these columns.
#
# Next, we can have a look at the encoded feature names.

# Before:
print(X.columns.to_list())
# After :
feature_names = sup_vec.get_feature_names()
print(', '.join(feature_names[:10]), '...')
print(len(feature_names))

# As we can see, it created a new column for each unique value.
# This is because we used SimilarityEncoder on the column "division",
# which was classified as a high cardinality string variable.
# (default values, see SuperVectorizer's docstring).
# In total, we have 1212 encoded columns.
#
# Finally, let's plot the features importance:
# Note: we will plot the features importances computed by the RandomForestRegressor,
# but you should use
# [permutation importances](https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html)
# instead (which are much more accurate).
# We chose the former over the later for the sake of performance.
import numpy as np
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
