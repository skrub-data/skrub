"""
Benchmark time consumption and scores for K-means and most frequent strategies.

We use the traffic_violations dataset to benchmark the different dimensionality
reduction strategies used in similarity encoding.

Parameters that are modified:
- Number of rows in datasets: 10k, 20k, 50k, 100k and nuniques.
- Hashing dimensions: 2 ** 14, 2 ** 16, 2 ** 18, 2 ** 20
- Ngram-range: (3, 3), (2, 4)
"""

# We filter out the warning asking us to specify the solver in the logistic_regression
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

from time import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import linear_model, model_selection
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from dirty_cat import SimilarityEncoder
from dirty_cat.datasets import fetch_traffic_violations


data = fetch_traffic_violations()

col_to_ohe = [
    "alcohol",
    "arrest_type",
    "belts",
    "commercial_license",
    "commercial_vehicle",
    "fatal",
    "gender",
    "hazmat",
    "property_damage",
    "race",
    "work_zone",
]

col_to_pass = ["year"]

col_to_enc = ["description"]

# Get the data and remove missing values:
X = data[1][col_to_enc + col_to_ohe + col_to_pass]
mask = X.isna()[col_to_pass].copy()
X.dropna(subset=col_to_pass, inplace=True)

y = data[2][~mask[mask.columns[0]]]

X.reset_index(inplace=True, drop=True)
y.reset_index(inplace=True, drop=True)

transformers = [
    ("one_hot", OneHotEncoder(sparse=False, handle_unknown="ignore"), col_to_ohe),
    ("pass", "passthrough", col_to_pass),
]


def benchmark(
    X_b,
    y_b,
    strat="k-means",
    limit=50000,
    n_proto=100,
    hash_dim=None,
    ngram_range=(3, 3),
):
    X_b = X_b[:limit].copy()
    y_b = y_b[:limit].copy()

    if strat == "k-means":
        sim_enc = SimilarityEncoder(
            ngram_range=ngram_range,
            categories="k-means",
            hashing_dim=hash_dim,
            n_prototypes=n_proto,
            random_state=3498,
        )
    else:
        sim_enc = SimilarityEncoder(
            ngram_range=ngram_range,
            categories="most_frequent",
            hashing_dim=hash_dim,
            n_prototypes=n_proto,
            random_state=3498,
        )

    column_trans = ColumnTransformer(
        transformers=transformers + [("sim_enc", sim_enc, ["description"])],
        remainder="drop",
    )

    t0 = time()
    X_enc = column_trans.fit_transform(X_b)
    t1 = time()
    t_score_1 = t1 - t0

    model = linear_model.LogisticRegression()

    t0 = time()
    m_score = model_selection.cross_val_score(model, X_enc, y_b, cv=20)
    t1 = time()
    t_score_2 = t1 - t0
    return t_score_1, m_score, t_score_2


def plot(bench, title=""):
    sns.set(style="ticks", palette="muted")
    hash_dims = ["Count", "2 ** 14", "2 ** 16", "2 ** 18", "2 ** 20"]
    scores = []
    vectorizer = []
    strategy = []

    for i, e in enumerate(bench):
        vectorizer.extend([hash_dims[i % 5]] * (2 * len(e[0][1])))
        strategy.extend(["k-means"] * len(e[0][1]))
        strategy.extend(["most-frequent"] * len(e[1][1]))
        scores.extend(e[0][1])
        scores.extend(e[1][1])

    df = pd.DataFrame(columns=["vectorizer", "strategy", "score"])
    df["vectorizer"] = vectorizer
    df["strategy"] = strategy
    df["score"] = scores

    first = plt.figure()
    ax = sns.boxplot(x="vectorizer", y="score", hue="strategy", data=df)
    ax.set(
        title=title,
        xlabel="Vectorizer used",
        ylabel="Mean score on 10 cross validations",
    )
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    first.tight_layout()

    vectorizer.clear()
    scores.clear()
    strategy.clear()
    times = []

    for i, e in enumerate(bench):
        vectorizer.extend([hash_dims[i % 5]] * 4)
        strategy.extend(["K-means vect", "K-means X-val", "MF vect", "MF X-val"])
        times.extend([e[0][0], e[0][2] / 20, e[1][0], e[1][2] / 20])

    df = pd.DataFrame(columns=["vectorizer", "strategy/operation", "time"])
    df["vectorizer"] = vectorizer
    df["strategy/operation"] = strategy
    df["time"] = times

    second = plt.figure()
    ax1 = sns.barplot(x="vectorizer", y="time", hue="strategy/operation", data=df)
    ax1.set(title=title, xlabel="Vectorizer used", ylabel="Time in seconds")
    ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    second.tight_layout()

    title = title.replace(" ", "_").replace(":", "-").replace(",", "_").lower()
    first.savefig(title + "_score.png")
    second.savefig(title + "_time.png")
    # first.show()
    # second.show(t)


def loop(proto):
    limits = sorted([X["description"].nunique(), 10000, 20000, 50000, 100000])
    hash_dims = [None, 2**14, 2**16, 2**18, 2**20]
    bench = list()
    ngram = [(3, 3), (2, 4)]
    for limit in limits:
        for r in ngram:
            for h in hash_dims:
                bench.append(
                    (
                        benchmark(
                            X,
                            y,
                            strat="k-means",
                            limit=limit,
                            n_proto=proto,
                            hash_dim=h,
                            ngram_range=r,
                        ),
                        benchmark(
                            X,
                            y,
                            strat="most-frequent",
                            limit=limit,
                            n_proto=proto,
                            hash_dim=h,
                            ngram_range=r,
                        ),
                    )
                )
            title = "N-gram range: %s, Rows: %d, Prototypes: %d, 20 CV" % (
                r.__str__(),
                limit,
                proto,
            )
            plot(bench, title)
            bench.clear()


if __name__ == "__main__":
    loop(100)
