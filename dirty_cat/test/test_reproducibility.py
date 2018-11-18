import numpy as np
import pandas as pd
from sklearn import pipeline, linear_model, model_selection
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from dirty_cat.datasets import fetch_traffic_violations
from dirty_cat.similarity_encoder import SimilarityEncoder


def get_result(random=None):
    data = fetch_traffic_violations()
    dfr = pd.read_csv(data['path'])

    transformers = [
        ('one_hot', OneHotEncoder(sparse=False, handle_unknown='ignore'),
         ['Alcohol',
          'Arrest Type',
          'Belts',
          'Commercial License',
          'Commercial Vehicle',
          'Fatal',
          'Gender',
          'HAZMAT',
          'Property Damage',
          'Race',
          'Work Zone']),
        ('pass', 'passthrough', ['Year']),
    ]

    df = dfr[:50000].copy()
    df = df.dropna(axis=0)
    df = df.reset_index()

    y = df['Violation Type']

    sim_enc = SimilarityEncoder(similarity='ngram', categories='most_frequent', n_prototypes=100, random_state=random)

    column_trans = ColumnTransformer(
        transformers=transformers + [('sim_enc', sim_enc, ['Description'])],
        remainder='drop'
    )

    X = column_trans.fit_transform(df)
    model = pipeline.Pipeline([('logistic', linear_model.LogisticRegression())])
    m_score = model_selection.cross_val_score(model, X, y)
    return X, m_score


mat1, scores1 = get_result(2454)
mat2, scores2 = get_result(2454)
assert (np.array_equal(mat1, mat2))
assert (np.array_equal(scores1, scores2))

random_state = np.random.RandomState(32456)
mat1, scores1 = get_result(random_state)
mat2, scores2 = get_result(random_state)
assert (np.array_equal(mat1, mat2))
assert (np.array_equal(scores1, scores2))
