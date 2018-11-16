import warnings
from time import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import pipeline, linear_model, model_selection
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from dirty_cat import SimilarityEncoder
from dirty_cat.datasets import fetch_traffic_violations

warnings.simplefilter(action='ignore', category=FutureWarning)

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


def benchmark_most_frequent(limit=50000, hash_dim=None, n_proto=100):
    df = dfr[:limit].copy()
    df = df.dropna(axis=0)
    df = df.reset_index()

    y = df['Violation Type']

    sim_enc = SimilarityEncoder(similarity='ngram', categories='most_frequent', n_prototypes=n_proto,
                                hashing_dim=hash_dim)

    column_trans = ColumnTransformer(
        transformers=transformers + [('sim_enc', sim_enc, ['Description'])],
        remainder='drop'
    )

    t0 = time()
    X = column_trans.fit_transform(df)
    t1 = time()
    t_score_1 = t1 - t0

    model = pipeline.Pipeline([('logistic', linear_model.LogisticRegression())])
    t0 = time()
    m_score = model_selection.cross_val_score(model, X, y, cv=10)
    t1 = time()
    t_score_2 = t1 - t0
    return t_score_1, m_score, t_score_2


def benchmark_k_means(limit=50000, n_proto=100, hash_dim=None, weights=False):
    df = dfr[:limit].copy()
    df = df.dropna(axis=0)
    df = df.reset_index()

    y = df['Violation Type']

    sim_enc = SimilarityEncoder(similarity='ngram', categories='k-means', hashing_dim=hash_dim, n_prototypes=n_proto,
                                sample_weight=weights)

    column_trans = ColumnTransformer(
        transformers=transformers + [('sim_enc', sim_enc, ['Description'])],
        remainder='drop'
    )

    t0 = time()
    X = column_trans.fit_transform(df)
    t1 = time()
    t_score_1 = t1 - t0

    model = pipeline.Pipeline([('logistic', linear_model.LogisticRegression())])

    t0 = time()
    m_score = model_selection.cross_val_score(model, X, y, cv=10)
    t1 = time()
    t_score_2 = t1 - t0
    return t_score_1, m_score, t_score_2


def plot(bench, title=''):
    hash_dims = ['Count', '2 ** 14', '2 ** 16', '2 ** 18', '2 ** 20']
    scores = []
    vectorizer = []
    strategy = []

    for i, e in enumerate(bench):
        vectorizer.extend([hash_dims[i % 5]] * (2 * len(e[0][1])))
        strategy.extend(['k-means'] * len(e[0][1]))
        strategy.extend(['most-frequent'] * len(e[1][1]))
        scores.extend(e[0][1])
        scores.extend(e[1][1])

    df = pd.DataFrame(columns=['vectorizer', 'strategy', 'score'])
    df['vectorizer'] = vectorizer
    df['strategy'] = strategy
    df['score'] = scores

    sns.set(style='ticks', palette='muted')
    sns.boxplot(x='vectorizer', y='score', hue='strategy', data=df)
    plt.ylabel("Mean score on 10 cross validations")
    plt.xlabel('Vectorizer used')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(title)
    plt.tight_layout()
    plt.show()

    vectorizer.clear()
    scores.clear()
    strategy.clear()
    times = []

    for i, e in enumerate(bench):
        vectorizer.extend([hash_dims[i % 5]] * 4)
        strategy.extend(
            ['K-means vect', 'K-means X-val', 'MF vect', 'MF X-val'])
        times.extend([e[0][0], e[0][2], e[1][0], e[1][2]])

    df = pd.DataFrame(columns=['vectorizer', 'strategy', 'time'])
    df['vectorizer'] = vectorizer
    df['strategy'] = strategy
    df['time'] = times

    sns.set(style='ticks', palette='muted')
    sns.barplot(x='vectorizer', y='time', hue='strategy', data=df)
    plt.ylabel("Time in seconds")
    plt.xlabel('Vectorizer used')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def loop(proto):
    limits = sorted([dfr['Description'].nunique(), 10000, 20000, 50000, 100000])
    hash_dims = [None, 2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20]
    weights = [False, True]
    bench = list()
    for limit in limits:
        for w in weights:
            for h in hash_dims:
                bench.append((benchmark_k_means(limit=limit, hash_dim=h, weights=w, n_proto=proto),
                              benchmark_most_frequent(limit=limit, n_proto=proto, hash_dim=h)))
            title = 'Weighted K-mean: %s, Rows: %d, Prototypes: %d' % (w.__str__(), limit, proto)
            plot(bench, title)
            bench.clear()


if __name__ == '__main__':
    loop(100)
