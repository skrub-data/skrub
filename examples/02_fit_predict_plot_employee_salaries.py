"""
Comparing encoders of a dirty categorical columns
==================================================

The column *Employee Position Title* of the dataset `employee salaries
<https://catalog.data.gov/dataset/employee-salaries-2016>`_ contains dirty categorical
data.

Here, we compare different categorical encodings for the dirty column to
predict the *Current Annual Salary*, using gradient boosted trees.
"""

################################################################################
# Data Importing and preprocessing
# --------------------------------
#
# We first download the dataset:
from dirty_cat.datasets import fetch_employee_salaries

info = fetch_employee_salaries()
print(info['description'])


################################################################################
# Then we load it:
import pandas as pd
df = pd.read_csv(info['path'], **info['read_csv_kwargs'])

################################################################################
# Now, let's carry out some basic preprocessing:
df['date_first_hired'] = pd.to_datetime(df['date_first_hired'])
df['year_first_hired'] = df['date_first_hired'].apply(lambda x: x.year)
# drop rows with NaN in gender
df.dropna(subset=['gender'], inplace=True)

target_column = 'current_annual_salary'
y = df[target_column].values.ravel()

#########################################################################
# Choosing columns
# -----------------
# For categorical columns that are supposed to be clean, it is "safe" to
# use one hot encoding to transform them:

clean_columns = {
    'gender': 'one-hot',
    'department_name': 'one-hot',
    'assignment_category': 'one-hot',
    'year_first_hired': 'numerical'}

#########################################################################
# We then choose the categorical encoding methods we want to benchmark
# and the dirty categorical variable:

encoding_methods = ['one-hot', 'target', 'similarity', 'minhash',
                    'gap']
dirty_column = 'employee_position_title'
#########################################################################


#########################################################################
# Creating a learning pipeline
# ----------------------------
# The encoders for both clean and dirty data are first imported:

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from dirty_cat import SimilarityEncoder, TargetEncoder, MinHashEncoder,\
    GapEncoder

# for scikit-learn 0.24 we need to require the experimental feature
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
# now you can import normally from ensemble
from sklearn.ensemble import HistGradientBoostingRegressor

encoders_dict = {
    'one-hot': OneHotEncoder(handle_unknown='ignore', sparse=False),
    'similarity': SimilarityEncoder(similarity='ngram'),
    'target': TargetEncoder(handle_unknown='ignore'),
    'minhash': MinHashEncoder(n_components=100),
    'gap': GapEncoder(n_components=100),
    'numerical': FunctionTransformer(None)}

# We then create a function that takes one key of our ``encoders_dict``,
# returns a pipeline object with the associated encoder,
# as well as a gradient-boosting regressor

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def assemble_pipeline(encoding_method):
    # static transformers from the other columns
    transformers = [(enc + '_' + col, encoders_dict[enc], [col])
                    for col, enc in clean_columns.items()]
    # adding the encoded column
    transformers += [(encoding_method, encoders_dict[encoding_method],
                      [dirty_column])]
    pipeline = Pipeline([
        # Use ColumnTransformer to combine the features
        ('union', ColumnTransformer(
            transformers=transformers,
            remainder='drop')),
        ('clf', HistGradientBoostingRegressor())
    ])
    return pipeline


#########################################################################
# Using each encoding for supervised learning
# --------------------------------------------
# Eventually, we loop over the different encoding methods,
# instantiate each time a new pipeline, fit it
# and store the returned cross-validation score:

from sklearn.model_selection import cross_val_score
import numpy as np

all_scores = dict()

for method in encoding_methods:
    pipeline = assemble_pipeline(method)
    scores = cross_val_score(pipeline, df, y)
    print('{} encoding'.format(method))
    print('r2 score:  mean: {:.3f}; std: {:.3f}\n'.format(
        np.mean(scores), np.std(scores)))
    all_scores[method] = scores

#########################################################################
# Plotting the results
# --------------------
# Finally, we plot the scores on a boxplot:

import seaborn
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 3))
ax = seaborn.boxplot(data=pd.DataFrame(all_scores), orient='h')
plt.ylabel('Encoding', size=20)
plt.xlabel('Prediction accuracy     ', size=20)
plt.yticks(size=20)
plt.tight_layout()

##########################################################################
# The clear trend is that encoders that use the string form
# of the category (similarity, minhash, and gap) perform better than
# those that discard it.
# 
# SimilarityEncoder is the best performer, but it is less scalable on big
# data than MinHashEncoder and GapEncoder. The most scalable encoder is
# the MinHashEncoder. GapEncoder, on the other hand, has the benefit that
# it provides interpretable features (see :ref:`sphx_glr_auto_examples_04_feature_interpretation_gap_encoder.py`)
