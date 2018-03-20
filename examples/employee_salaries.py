"""
Predicting the salary of employees
==================================

Benchmark of encoders for the "employee_salaries" dataset.

"""

import numpy as np
from scipy import sparse

import pandas as pd

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from dirty_cat import datasets
from dirty_cat import SimilarityEncoder, TargetEncoder


# encoding methods
encoder_dict = {
    'one-hot': OneHotEncoder(handle_unknown='ignore'),
    'similarity': SimilarityEncoder(similarity='ngram',
                                    handle_unknown='ignore'),
    'target': TargetEncoder(handle_unknown='ignore'),
    'num': FunctionTransformer(None)
    }

data_file = datasets.fetch_employee_salaries()

for method in ['one-hot', 'target', 'similarity']:
    # Load the data
    df = pd.read_csv(data_file).astype(str)
    df['Current Annual Salary'] = [float(s[1:]) for s
                                   in df['Current Annual Salary']]
    df['Year First Hired'] = [int(s.split('/')[-1])
                              for s in df['Date First Hired']]

    target_column = 'Current Annual Salary'
    y = df[target_column].values.ravel()

    # Transform the data into a numerical matrix
    encoder_type = {
        'one-hot': ['Gender', 'Department Name', 'Assignment Category'],
        'num': ['Year First Hired']
        }
    try:
        encoder_type[method].append('Employee Position Title')
    except KeyError:
        encoder_type[method] = ['Employee Position Title']

    # OneHotEncoder needs numerical data, hence we first use LabelEncoder
    label_encoder = LabelEncoder()
    df[encoder_type['one-hot']] = df[
        encoder_type['one-hot']].apply(label_encoder.fit_transform)

    cv = KFold(n_splits=5, random_state=12, shuffle=True)

    scores = []
    for train_index, test_index in cv.split(df, df[target_column]):
        y_train = y[train_index]
        X_train = [
            encoder_dict[encoder].fit_transform(
                df.loc[train_index, encoder_type[encoder]
                       ].values.reshape(len(train_index), -1), y_train)
            for encoder in encoder_type]
        X_train = sparse.hstack(X_train).toarray()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)

        y_test = y[test_index]
        X_test = [
            encoder_dict[encoder].transform(
                 df.loc[test_index, encoder_type[encoder]
                        ].values.reshape(len(test_index), -1))
            for encoder in encoder_type]
        X_test = sparse.hstack(X_test).toarray()
        X_test = scaler.transform(X_test)

        # Now predict the salary of each worker
        classifier = RidgeCV()
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        scores.append(score)

    print('%s encoding' % method)
    print('R^2 score:  mean: %.3f; std: %.3f\n'
          % (np.mean(scores), np.std(scores)))
