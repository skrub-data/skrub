"""
Using target encoder with K-fold cross-validation
===================================================

Here we discuss how to apply efficiently the :class:`TargetEncoder`. If the
dataset is large enough, it may be useful to split the data and encode
values on a subset. That way, overfitting is avoided and we may get
better test results. This can be done easily using the ``cross_val``
parameter of the :class:`TargetEncoder`.


It is also possible to choose the number of outer and inner folds
into which the data will be splitted (using the ``n_folds`` and
``n_inner_folds`` parameters).

"""

###############################################################################
# Data Importing and preprocessing
# --------------------------------
#
# We first get the dataset:
import pandas as pd
from dirty_cat.datasets import fetch_road_safety

road_safety = fetch_road_safety()

print(road_safety.description)

# X, our explanatory variables
X = road_safety.X
X
# and y, our target column (the driver's sex)
y = road_safety.y

# Now, let's carry out some basic preprocessing:

# Keep only columns that will be used:
col_to_use = ['Age_of_Driver', 'Age_of_Vehicle', 'Day_of_Week', 'Speed_limit', 'Weather_Conditions', 'Local_Authority_(Highway)']

X = X[col_to_use]

# Drop the lines that contained missing values in X and y
for col in col_to_use:
    mask = X.isna()[col]
    X.dropna(subset=[col], inplace=True)
    y = y[~mask]

X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

###############################################################################
# Assembling a machine-learning pipeline that encodes the data
# ============================================================
#
# The pipeline
# ---------------------
# Remark: an encoder is used to turn a categorical column into a numerical
# representation

# We create our encoders. In this example we will
# compare the ``OneHotEncoder`` to the ``TargetEncoder``,
# used with or without cross-validation encoding.
from sklearn.preprocessing import OneHotEncoder
one_hot = OneHotEncoder(handle_unknown='ignore', sparse=False)

# To be noted is that the type of the problem needs to
#  be specified to the ``TargetEncoder``.
# In this case we are in a binary classification problem,
# so we will use ``clf_type='binary-clf'``.
from dirty_cat import TargetEncoder
target = TargetEncoder(clf_type='binary-clf', handle_unknown='ignore', cross_val=False)
target_cv = TargetEncoder(clf_type='binary-clf', handle_unknown='ignore', cross_val=True, n_folds=4, n_inner_folds=3)

encoders = {
    'one-hot': one_hot,
    'target': target,
    'target-cv': target_cv
}

# We assemble these to apply them to the relevant columns.
# The ColumnTransformer is created by specifying a set of transformers
# alongside with the column names on which each must be applied

from sklearn.compose import make_column_transformer

# We will use a HistGradientBoostingClassifier :
from sklearn.ensemble import HistGradientBoostingClassifier

# We then create a pipeline chaining our encoders to a learner
from sklearn.pipeline import make_pipeline

# Dirty-category encoding
# -----------------------
#
# We now loop over the different encoding methods,
# instantiate a new |Pipeline| each time, fit it
# and store the returned cross-validation score:

from sklearn.model_selection import cross_validate
import numpy as np

all_scores = dict()

for name, method in encoders.items():
    encoder = make_column_transformer(
        (one_hot, ['Day_of_Week', 'Weather_Conditions', 'Speed_limit']),
        ('passthrough', ['Age_of_Driver', 'Age_of_Vehicle']),
        # Last but not least, our dirty column
        (method, ['Local_Authority_(Highway)']),
        remainder='drop',
    )

    pipeline = make_pipeline(encoder, HistGradientBoostingClassifier())
    scores = cross_validate(pipeline, X, y)
    test_scores = scores['test_score']
    print(f'{name} encoding')
    print(f'r2 score:  mean: {np.mean(test_scores):.3f}; '
          f'std: {np.std(test_scores):.3f}\n')
    all_scores[name] = scores

# The results show that the :class:`TargetEncoder` performs best
# if the data are split into folds that will then determine the
# encoded values. It outperforms also the ``OneHotEncoder``.

###############################################################################
# Plot a summary figure
# ---------------------
import seaborn
import matplotlib.pyplot as plt

fit_times = dict()
test_results = dict()
for enc in encoders.keys():
    fit_times[enc] = all_scores[enc]['fit_time']
    test_results[enc] = all_scores[enc]['test_score']

_, (ax1, ax2) = plt.subplots(nrows=2, figsize=(4, 3))

seaborn.boxplot(data=pd.DataFrame(test_results), orient='h', ax=ax1)
ax1.set_xlabel('Prediction accuracy', size=16)
[t.set(size=16) for t in ax1.get_yticklabels()]

seaborn.boxplot(data=pd.DataFrame(fit_times), orient='h', ax=ax2)
ax2.set_xlabel('Computation time', size=16)
[t.set(size=16) for t in ax2.get_yticklabels()]
plt.tight_layout()

print(test_results)

# We can observe the fit times and the test scores.
# It is clear that the ``TargetEncoder`` with cross-validation has
# much better, and much less variable, prediction scores. This is due
# to the better generalization of the model, that avoids overfitting.
# Finally, the fitting time of the ``TargetEncoder`` with K-fold splitting
# is somewhat slower than without, but much faster than
# with the ``OneHotEncoder``.
