import copy

from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import skrub
from skrub._expressions._estimator import _SharedDict


def get_classifier(c_steps=None):
    X, y = skrub.X(), skrub.y()
    return (
        X.skb.apply(SelectKBest(), y=y)
        .skb.apply(StandardScaler())
        .skb.apply(
            LogisticRegression(
                **skrub.choose_float(0.01, 1.0, log=True, n_steps=c_steps, name="C")
            ),
            y=y,
        )
    )


def test_fit_predict():
    pred = get_classifier()
    estimator = pred.skb.get_estimator()
    X_train, X_test, y_train, y_test = train_test_split(
        *make_classification(random_state=0)
    )
    estimator.fit({"X": X_train, "y": y_train})

    # Note that y is missing from the environment here. It is not needed for
    # predict (or transform, score etc.)
    predicted = estimator.predict({"X": X_test})

    assert accuracy_score(y_test, predicted) > 0.8


def test_cross_validate():
    pred = get_classifier()
    X, y = make_classification(random_state=0)
    score = pred.skb.cross_validate({"X": X, "y": y})["test_score"]
    assert len(score) == 5
    assert 0.8 < score.mean() < 0.9


def test_search():
    pred = get_classifier()
    X, y = make_classification(random_state=0)
    search = pred.skb.get_randomized_search(n_iter=3, random_state=0)
    search.fit({"X": X, "y": y})
    assert 0.8 < search.results_["mean_test_score"].iloc[0] < 0.9


def test_nested_cv():
    pred = get_classifier()
    X, y = make_classification(random_state=0)
    search = pred.skb.get_randomized_search(n_iter=3, random_state=0)
    score = skrub.cross_validate(search, {"X": X, "y": y})["test_score"]
    assert len(score) == 5
    assert 0.8 < score.mean() < 0.9


#
# misc utils from the _estimator module
#


def test_shared_dict():
    d = _SharedDict({"a": 0})
    assert clone(d) is d
    assert copy.deepcopy(d) is d
