"""
Predicting the salary of employees
==================================

Let's have a look on how using similarity encoding instead of
more traditional categorical encoding like one-hot encoding can affect
prediction performance.

The structure of this example will run as follows:

* in the employee_salaries dataset we will first create a learning problem by 
 chosing a column to predict

* then we will benchmark the different kind of encodings (one-hot, similarity \
encoding, target encoding) for one categoricl variable
by fitting the target using their respective output \
and a Ridge Regression


The learning problem consists of prediciting the column 'Current Annual Salary'\
depending on a mix of clean columns and on dirty column \
We choose to benchmark different categorical encodings for \
the dirty column 'Employee Position Title', that contains \
dirty categorical data.

**Warning: this example is using the master branch of scikit-learn**

"""

################################################################################
# Data Importing and preprocessing
# ---------------------
# we first import the datataset 'employee_salaries'
import pandas as pd
from dirty_cat.datasets import fetch_employee_salaries

description = fetch_employee_salaries()
df = pd.read_csv(description['path']).astype(str)

################################################################################
# and carry out some basic preprocessing:
df['Current Annual Salary'] = df['Current Annual Salary'].str.strip('$').astype(
    float)
df['Date First Hired'] = pd.to_datetime(df['Date First Hired'])
df['Year First Hired'] = df['Date First Hired'].apply(lambda x: x.year)

target_column = 'Current Annual Salary'
y = df[target_column].values.ravel()

#########################################################################
# Choosing clean columns
# ----------------------
# the other column are supposed clean, so it is 'safe' to use
# one hot encoding to transform them

clean_columns = {
    'Gender': 'one-hot',
    'Department Name': 'one-hot',
    'Assignment Category': 'one-hot',
    'Year First Hired': 'num'}

#########################################################################
# We then choose  which categorical encoding methods to benchmark:
encoding_methods = ['one-hot', 'target', 'similarity']
dirty_column = 'Employee Position Title'
#########################################################################


#########################################################################
# Creating a model fitting pipeline
# ------------------------
# the encoders for both clean and dirty data are first imported:
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import CategoricalEncoder
from dirty_cat import SimilarityEncoder, TargetEncoder

encoders_dict = {
    'one-hot': CategoricalEncoder(handle_unknown='ignore',
                                  encoding='onehot-dense'),
    'similarity': SimilarityEncoder(similarity='ngram',
                                    handle_unknown='ignore'),
    'target': TargetEncoder(handle_unknown='ignore'),
    'num': FunctionTransformer(None)}

# we create a function that takes one key of our encoders_dict,
# returns a encoding+fitting pipeline with the associated encoder,
# as well as a Scaler and a RidgeCV regressor:
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def make_pipeline(encoding_method):
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
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', RidgeCV())
    ])
    return pipeline


#########################################################################
# Fitting each encoding methods with a RidgeCV
# --------------------------------------------
# eventually, we loop over the different encoding methods,
# instanciate each time a new pipeline, fit it
# and and store the returned cross-validation score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score
import numpy as np

all_scores = []

cv = KFold(n_splits=5, random_state=12, shuffle=True)
scoring = 'r2'
for method in encoding_methods:
    pipeline = make_pipeline(method)
    scores = cross_val_score(pipeline, df, y, cv=cv, scoring=scoring)
    print('{} encoding'.format(method))
    print('{} score:  mean: {:.3f}; std: {:.3f}\n'.format(
        scoring, np.mean(scores), np.std(scores)))
    all_scores.append(scores)

#########################################################################
# Plotting the results
# --------------------
# plotting the scores on a boxplot, we get:
import matplotlib.pyplot as plt

f, ax = plt.subplots()
ax.boxplot(all_scores)
ax.set_xticklabels(encoding_methods)



