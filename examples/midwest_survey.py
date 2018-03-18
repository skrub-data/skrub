"""
Semantic variation in the "Midwest"
"""

import numpy as np
from scipy import sparse

from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import cross_val_score

from dirty_cat.datasets import fetch_midwest_survey
from dirty_cat.similarity_encoder import SimilarityEncoder


# encoding methods
encoder_dict = {
    'one-hot': OneHotEncoder(handle_unknown='ignore'),
    'similarity': SimilarityEncoder(similarity='ngram',
                                    handle_unknown='ignore'),
    'num': FunctionTransformer(None)
    }

for method in ['one-hot', 'similarity']:
    df = fetch_midwest_survey().astype(str)

    target_column = 'Location (Census Region)'
    y = df[target_column].values.ravel()

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
    # LabelEncoder before using OneHotEncoder
    label_encoder = LabelEncoder()
    onehot_columns = [col for col, enc in feature_columns if enc == 'one-hot']
    df[onehot_columns] = df[onehot_columns].apply(label_encoder.fit_transform)

    X = [encoder_dict[encoder].fit_transform(df[column].values.reshape(-1, 1))
         for column, encoder in feature_columns]
    X = sparse.hstack(X)

    pipeline = Pipeline([
                         ('scaler', StandardScaler(with_mean=False)),
                         ('classifier', RandomForestClassifier(
                             random_state=5))
                         ])

    print(method)
    scores = cross_val_score(pipeline, X, y, cv=5)
    print('Accuracy:  mean: %.3f; std: %.3f\n'
          % (np.mean(scores), np.std(scores)))
