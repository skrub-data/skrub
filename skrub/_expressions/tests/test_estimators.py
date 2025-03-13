import copy
import io

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import skrub
from skrub._expressions._estimator import _SharedDict


def get_data():
    X, y = make_classification(random_state=0)
    return {"X": X, "y": y}


@skrub.deferred
def _read_csv(csv_str):
    return pd.read_csv(io.BytesIO(csv_str), encoding="utf-8")


def _to_csv(df):
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue()


def get_unprocessed_data():
    X, y = make_classification(random_state=0)
    X = pd.DataFrame(X, columns=[f"c_{i}" for i in range(X.shape[1])]).assign(
        ID=np.arange(X.shape[0])
    )
    y = pd.DataFrame({"target": y, "ID": X["ID"]}).sample(frac=1).reset_index()
    return {"X": _to_csv(X), "y": _to_csv(y)}


def _make_classifier(X, y, *, discrete_grid):
    c_steps = 2 if discrete_grid else None
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


def get_classifier(discrete_grid=False):
    return _make_classifier(skrub.X(), skrub.y(), discrete_grid=discrete_grid)


def get_classifier_for_unprocessed_data(discrete_grid=False):
    X, y = skrub.var("X"), skrub.var("y")
    X = _read_csv(X)
    y = _read_csv(y)
    y = X.merge(y, on="ID", how="left")["target"].skb.mark_as_y()
    X = X.drop(columns="ID").skb.mark_as_X()
    return _make_classifier(X, y, discrete_grid=discrete_grid)


def get_classifier_and_data(processed=True, discrete_grid=False):
    if processed:
        return get_classifier(discrete_grid=discrete_grid), get_data()
    return (
        get_classifier_for_unprocessed_data(discrete_grid=discrete_grid),
        get_unprocessed_data(),
    )


def test_fit_predict():
    pred, data = get_classifier_and_data()
    estimator = pred.skb.get_estimator()
    X_train, X_test, y_train, y_test = train_test_split(
        data["X"], data["y"], shuffle=False
    )
    estimator.fit({"X": X_train, "y": y_train})

    # Note that y is missing from the environment here. It is not needed for
    # predict (or transform, score etc.)
    predicted = estimator.predict({"X": X_test})

    assert 0.75 < accuracy_score(y_test, predicted)


@pytest.mark.parametrize("processed", [True, False])
def test_cross_validate(processed):
    pred, data = get_classifier_and_data(processed=processed)
    score = pred.skb.cross_validate(data)["test_score"]
    assert len(score) == 5
    assert 0.75 < score.mean() < 0.9


@pytest.mark.parametrize("processed", [True, False])
def test_search(processed):
    pred, data = get_classifier_and_data(processed=processed)
    search = pred.skb.get_randomized_search(n_iter=3, random_state=0)
    search.fit(data)
    assert 0.75 < search.results_["mean_test_score"].iloc[0] < 0.9


@pytest.mark.parametrize("processed", [True, False])
def test_nested_cv(processed):
    pred, data = get_classifier_and_data(processed=processed)
    search = pred.skb.get_randomized_search(n_iter=3, random_state=0)
    score = skrub.cross_validate(search, data)["test_score"]
    assert len(score) == 5
    assert 0.75 < score.mean() < 0.9


#
# misc utils from the _estimator module
#


def test_shared_dict():
    d = _SharedDict({"a": 0})
    assert clone(d, safe=False) is d
    assert copy.deepcopy(d) is d
