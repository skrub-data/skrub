"""
base example script. the midwest survey datasets will be included in the
package because of its small size (500kb)
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
from sklearn.model_selection import cross_val_score

from dirty_cat.datasets import fetch_midwest_survey
from dirty_cat.similarity_encoder import SimilarityEncoder


# encoding methods
encoder_dict = {
    'onehot': OneHotEncoder(handle_unknown='ignore'),
    'similarity': SimilarityEncoder(similarity='ngram',
                                    handle_unknown='ignore'),
    'num': FunctionTransformer(None)
    }


for method in ['onehot', 'similarity']:
    df = fetch_midwest_survey().astype(str)

    target_column = 'Location (Census Region)'
    y = df[target_column].values.ravel()

    feature_columns = [
        ('In your own words, what would you call the part of the country '
         'you live in now?', method),
        ('Wyoming in MW?', 'onehot'),
        ('Age', 'onehot')
        ]
    # LabelEncoder before using OneHotEncoder
    label_encoder = LabelEncoder()
    onehot_columns = [col for col, enc in feature_columns if enc == 'onehot']
    df[onehot_columns] = df[onehot_columns].apply(label_encoder.fit_transform)

    X = [encoder_dict[encoder].fit_transform(df[column].values.reshape(-1, 1))
         for column, encoder in feature_columns]
    X = sparse.hstack(X)

    pipeline = Pipeline([
                         ('scaler', StandardScaler(with_mean=False)),
                         ('classifier', RandomForestClassifier(
                             random_state=12))
                         ])

    print(method)
    scores = cross_val_score(pipeline, X, y, cv=5)
    print('Accuracy:  mean: %.3f; std: %.3f\n'
          % (np.mean(scores), np.std(scores)))
