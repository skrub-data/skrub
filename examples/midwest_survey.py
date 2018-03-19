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
from sklearn.model_selection import cross_val_score

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
    feature_columns = [
        ('In your own words, what would you call the part of the country '
         'you live in now?', method),
        ('Personally identification as a Midwesterner?', 'one-hot'),
        ('Illinois in MW?', 'one-hot'),
        ('Indiana in MW?', 'one-hot'),
        ('Kansas in MW?', 'one-hot'),
        ('Iowa in MW?', 'one-hot'),
        ('Michigan in MW?', 'one-hot'),
        ('Minnesota in MW?', 'one-hot'),
        ('Missouri in MW?', 'one-hot'),
        ('Nebraska in MW?', 'one-hot'),
        ('North Dakota in MW?', 'one-hot'),
        ('Ohio in MW?', 'one-hot'),
        ('South Dakota in MW?', 'one-hot'),
        ('Wisconsin in MW?', 'one-hot'),
        ('Arkansas in MW?', 'one-hot'),
        ('Colorado in MW?', 'one-hot'),
        ('Kentucky in MW?', 'one-hot'),
        ('Oklahoma in MW?', 'one-hot'),
        ('Pennsylvania in MW?', 'one-hot'),
        ('West Virginia in MW?', 'one-hot'),
        ('Montana in MW?', 'one-hot'),
        ('Wyoming in MW?', 'one-hot'),
        ('Gender', 'one-hot'),
        ('Age', 'one-hot'),
        ('Household Income', 'one-hot'),
        ('Education', 'one-hot'),
        ]
    # OneHotEncoder needs numerical data, hence we first use LabelEncoder
    label_encoder = LabelEncoder()
    onehot_columns = [col for col, enc in feature_columns if enc == 'one-hot']
    df[onehot_columns] = df[onehot_columns].apply(label_encoder.fit_transform)

    X = [encoder_dict[encoder].fit_transform(df[column].values.reshape(-1, 1))
         for column, encoder in feature_columns]
    X = sparse.hstack(X)

    # Now predict whether or not each row is about the midwest
    classifier = RandomForestClassifier(random_state=5)
    print('%s encoding' % method)
    scores = cross_val_score(classifier, X, y, cv=5)
    print('Accuracy:  mean: %.3f; std: %.3f\n'
          % (np.mean(scores), np.std(scores)))
