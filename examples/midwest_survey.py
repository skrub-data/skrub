"""
Semantic variation in the "Midwest"
===================================

Benchmark of encoders for the midwest_survey dataset: the data comprises
an open-ended question, on which one-hot encoding does not work well.

Similarity encoding on this column gives much improved performance.

"""

import numpy as np
from scipy import sparse

import pandas as pd

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

from dirty_cat import datasets
from dirty_cat import SimilarityEncoder


# encoding methods
encoder_dict = {
    'one-hot': OneHotEncoder(handle_unknown='ignore'),
    'similarity': SimilarityEncoder(similarity='ngram',
                                    handle_unknown='ignore'),
    'num': FunctionTransformer(None)
    }

data_file = datasets.fetch_midwest_survey()

for method in ['one-hot', 'similarity']:
    # Load the data
    df = pd.read_csv(data_file).astype(str)

    target_column = 'Location (Census Region)'
    y = df[target_column].values.ravel()

    # Transform the data into a numerical matrix
    encoder_type = {
        'one-hot': [
            'Personally identification as a Midwesterner?',
            'Illinois in MW?',
            'Indiana in MW?',
            'Kansas in MW?',
            'Iowa in MW?',
            'Michigan in MW?',
            'Minnesota in MW?',
            'Missouri in MW?',
            'Nebraska in MW?',
            'North Dakota in MW?',
            'Ohio in MW?',
            'South Dakota in MW?',
            'Wisconsin in MW?',
            'Arkansas in MW?',
            'Colorado in MW?',
            'Kentucky in MW?',
            'Oklahoma in MW?',
            'Pennsylvania in MW?',
            'West Virginia in MW?',
            'Montana in MW?',
            'Wyoming in MW?',
            'Gender',
            'Age',
            'Household Income',
            'Education']
        }
    try:
        encoder_type[method].append(
            'In your own words, what would you call the part of the country '
            'you live in now?')
    except KeyError:
        encoder_type[method] = ('In your own words, what would you call the '
                               'part of the country you live in now?')

    # OneHotEncoder needs numerical data, hence we first use LabelEncoder
    label_encoder = LabelEncoder()
    df[encoder_type['one-hot']] = df[
        encoder_type['one-hot']].apply(label_encoder.fit_transform)

    cv = StratifiedKFold(n_splits=3, random_state=12, shuffle=True)

    scores = []
    for train_index, test_index in cv.split(df, df[target_column]):
        X_train = [
            encoder_dict[encoder].fit_transform(
            df.loc[train_index, encoder_type[encoder]
                   ].values.reshape(len(train_index), -1))
            for encoder in encoder_type]
        X_train = sparse.hstack(X_train)
        y_train = y[train_index]

        X_test = [
            encoder_dict[encoder].transform(
            df.loc[test_index, encoder_type[encoder]
                   ].values.reshape(len(test_index), -1))
            for encoder in encoder_type]
        X_test = sparse.hstack(X_test)
        y_test = y[test_index]
        X_test.shape
        X_train.shape
        
        # Now predict the census region of each participant
        classifier = RandomForestClassifier(random_state=5)
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        scores.append(score)

    print('%s encoding' % method)
    print('Accuracy score:  mean: %.3f; std: %.3f\n'
          % (np.mean(scores), np.std(scores)))
