"""
Predicting the salary of employees
==================================

The `employee salaries <https://catalog.data.gov/dataset/employee-salaries-2016>`_
dataset contains information
about annual salaries (year 2016) for more than 9,000 employees of the 
Montgomery County (Maryland, US). In this example, we are interested
in predicting the column *Current Annual Salary*
depending on a mix of clean columns and on dirty column.
We choose to benchmark different categorical encodings for
the dirty column *Employee Position Title*, that contains
dirty categorical data.

**Warning: this example is using the master branch of scikit-learn**

"""

################################################################################
# Data Importing and preprocessing
# --------------------------------
# We first import the datataset:
import pandas as pd
from dirty_cat.datasets import fetch_employee_salaries

employee_salaries = fetch_employee_salaries()
df = pd.read_csv(employee_salaries['path']).astype(str)
print(employee_salaries['description'])
################################################################################
# and carry out some basic preprocessing:

df['Current Annual Salary'] = df['Current Annual Salary'].str.strip('$').astype(
    float)
df['Date First Hired'] = pd.to_datetime(df['Date First Hired'])
df['Year First Hired'] = df['Date First Hired'].apply(lambda x: x.year)

target_column = 'Current Annual Salary'
y = df[target_column].values.ravel()

#########################################################################
# Choosing columns
# -----------------
# For categorical columns that are supossed to be clean, it is "safe" to
# use one hot encoding to transform them:

clean_columns = {
    'Gender': 'one-hot',
    'Department Name': 'one-hot',
    'Assignment Category': 'one-hot',
    'Year First Hired': 'numerical'}

#########################################################################
# We then choose the categorical encoding methods we want to benchmark
# and the dirty categorical variable:

encoding_methods = ['one-hot', 'target', 'similarity']
dirty_column = 'Employee Position Title'
#########################################################################


#########################################################################
# Creating a learning pipeline
# ----------------------------
# The encoders for both clean and dirty data are first imported:

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import CategoricalEncoder
from dirty_cat import SimilarityEncoder, TargetEncoder

encoders_dict = {
    'one-hot': CategoricalEncoder(handle_unknown='ignore',
                                  encoding='onehot-dense'),
    'similarity': SimilarityEncoder(similarity='ngram',
                                    handle_unknown='ignore'),
    'target': TargetEncoder(handle_unknown='ignore'),
    'numerical': FunctionTransformer(None)}

# We then create a function that takes one key of our ``encoders_dict``,
# returns a pipeline object with the associated encoder,
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
# Eventually, we loop over the different encoding methods,
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
# Finally, we plot the scores on a boxplot:
import matplotlib.pyplot as plt

f, ax = plt.subplots()
ax.boxplot(all_scores, vert=False)
ax.set_yticklabels(encoding_methods)



