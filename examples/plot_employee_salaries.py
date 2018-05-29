"""
Predicting the salary of employees
==================================

Now that we understand how the similarity encoder works, let's 
see how this can affect prediction performance.

The structure of this example will run as follows:
* in the employee_salaries dataset we will first create a learning problem by 
 chosing a column to predict
* then we will benchmark the different kind of encodings (one-hot, similarity encoding, target encoding) 
by fitting the target with their respective output with a Ridge Regression


"""

#########################################################################
# 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import CategoricalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from dirty_cat.datasets import fetching
from dirty_cat import SimilarityEncoder, TargetEncoder
#########################################################################
# we first define the encoding methods we will benchmark
# encoding methods
encoders_dict = {
    'one-hot': CategoricalEncoder(handle_unknown='ignore',
                                  encoding='onehot-dense'),
    'similarity': SimilarityEncoder(similarity='ngram',
                                    handle_unknown='ignore'),
    'target': TargetEncoder(handle_unknown='ignore'),
    'num': FunctionTransformer(None)}
#########################################################################
#we then get the data, and define a target column we will try to predict, 
# as well as a dirty colum we will encode with the different methods.
# the rest will have a standard encoding
data_file = fetching.fetch_employee_salaries()
data_file=os.path.join(data_file,'rows.csv')
df = pd.read_csv(data_file).astype(str)
df['Current Annual Salary'] = [float(s[1:]) for s
                               in df['Current Annual Salary']]
df['Year First Hired'] = [int(s.split('/')[-1])
                          for s in df['Date First Hired']]

target_column = 'Current Annual Salary'
y = df[target_column].values.ravel()

encoded_column='Employee Position Title'


#########################################################################
# the other column are supposed clean, so it is 'safe' to use
# one hot encoding to transform them


columns_to_encode = {
    'one-hot': ['Gender', 'Department Name', 'Assignment Category'],
    'num': ['Year First Hired']}
#########################################################################
# let's then define a quick fit_score function that will re-use for 
# each encoder
def fit_score(df,method):
    #all the other columns are encoded 
    columns_to_encode.setdefault(method, [])
    columns_to_encode[method].append(encoded_column)

    pipeline = Pipeline([
        # Use ColumnTransformer to combine the features
        ('union', ColumnTransformer(
            [(e, encoders_dict[e], columns_to_encode[e])
             for e in columns_to_encode])),
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', RidgeCV())
        ])
    cv = KFold(n_splits=5, random_state=12, shuffle=True)
    scoring = 'r2'
    scores = cross_val_score(pipeline, df, y, cv=cv, scoring=scoring)
    print('%s encoding' % method)
    print('%s score:  mean: %.3f; std: %.3f\n' % (scoring,np.mean(scores), np.std(scores)))
    return scores

#########################################################################
#then, let's run this functino to get some scores for the three encodings
scores=[]
encoding_methods=['one-hot', 'target', 'similarity']
for method in encoding_methods:
    scores.append(fit_score(df,method))

#########################################################################
# plotting the scores on a boxplot, we get:
f,ax=plt.subplots()
ax.set_xticks(range(len(encoding_methods)))
ax.set_xticklabels(encoding_methods)
ax.boxplot(scores)

