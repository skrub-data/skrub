import copy
import io
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import skrub
from skrub._expressions._estimator import _SharedDict

#
# testing utils
#


@skrub.deferred
def _read_csv(csv_str):
    return pd.read_csv(io.BytesIO(csv_str), encoding="utf-8")


def _to_csv(df):
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8")
    return buf.getvalue()


def _make_classifier(X, y):
    return (
        X.skb.apply(SelectKBest(), y=y)
        .skb.apply(StandardScaler())
        .skb.apply(
            LogisticRegression(
                **skrub.choose_float(0.01, 1.0, log=True, n_steps=4, name="C")
            ),
            y=y,
        )
    )


def _simple_data():
    X, y = make_classification(random_state=0)
    return {"X": X, "y": y}


def _simple_data_classifier():
    return _make_classifier(skrub.X(), skrub.y())


def _unprocessed_data():
    X, y = make_classification(random_state=0)
    X = pd.DataFrame(X, columns=[f"c_{i}" for i in range(X.shape[1])]).assign(
        ID=np.arange(X.shape[0])
    )
    y = pd.DataFrame({"target": y, "ID": X["ID"]}).sample(frac=1).reset_index()
    return {"X": _to_csv(X), "y": _to_csv(y)}


def _unprocessed_data_classifier():
    X, y = skrub.var("X"), skrub.var("y")
    X = _read_csv(X)
    y = _read_csv(y)
    y = X.merge(y, on="ID", how="left")["target"].skb.mark_as_y()
    X = X.drop(columns="ID").skb.mark_as_X()
    return _make_classifier(X, y)


def get_expression_and_data(data_kind):
    assert data_kind in ["simple", "unprocessed"]
    if data_kind == "simple":
        return _simple_data_classifier(), _simple_data()
    return _unprocessed_data_classifier(), _unprocessed_data()


@pytest.fixture(params=["simple", "unprocessed"])
def _expression_and_data(request):
    data_kind = request.param
    expr, data = get_expression_and_data(data_kind)
    return {"data_kind": data_kind, "expression": expr, "data": data}


@pytest.fixture
def expression(_expression_and_data):
    return _expression_and_data["expression"]


@pytest.fixture
def data(_expression_and_data):
    return _expression_and_data["data"]


@pytest.fixture
def data_kind(_expression_and_data):
    return _expression_and_data["data_kind"]


@pytest.fixture(params=[None, 1, 2])
def n_jobs(request):
    return request.param


#
# fit, predict, param search & (possibly nested) cross-validation
#


def test_fit_predict():
    expression, data = get_expression_and_data("simple")
    estimator = expression.skb.get_estimator()
    X_train, X_test, y_train, y_test = train_test_split(
        data["X"], data["y"], shuffle=False
    )
    estimator.fit({"X": X_train, "y": y_train})

    # Note that y is missing from the environment here. It is not needed for
    # predict (or transform, score etc.)
    predicted = estimator.predict({"X": X_test})

    assert 0.75 < accuracy_score(y_test, predicted)


def test_cross_validate(expression, data, n_jobs):
    score = expression.skb.cross_validate(data, n_jobs=n_jobs)["test_score"]
    assert len(score) == 5
    assert 0.75 < score.mean() < 0.9


def test_randomized_search(expression, data, n_jobs):
    search = expression.skb.get_randomized_search(
        n_iter=3, n_jobs=n_jobs, random_state=0
    )
    search.fit(data)
    assert 0.75 < search.results_["mean_test_score"].iloc[0] < 0.9


def test_grid_search(expression, data, n_jobs):
    search = expression.skb.get_grid_search(n_jobs=n_jobs)
    search.fit(data)
    assert 0.75 < search.results_["mean_test_score"].iloc[0] < 0.9


def test_nested_cv(expression, data, data_kind, n_jobs, monkeypatch):
    search = expression.skb.get_randomized_search(
        n_iter=3, n_jobs=n_jobs, random_state=0
    )
    mock = Mock(side_effect=pd.read_csv)
    monkeypatch.setattr(pd, "read_csv", mock)

    score = skrub.cross_validate(search, data, n_jobs=n_jobs)["test_score"]

    # when data is loaded from csv we check that the caching results in
    # read_csv being called exactly twice (we have 2 different 'files' to
    # load).
    assert mock.call_count == (2 if data_kind == "unprocessed" else 0)

    assert len(score) == 5
    assert 0.75 < score.mean() < 0.9


@skrub.deferred
def _count_passthrough(value, counter, name):
    counter["count"] = counter.get("count", 0) + 1
    return value


@skrub.deferred
def _load_data():
    X, y = make_classification(random_state=0)
    return {"features": X, "target": y}


def test_caching():
    data = _load_data()
    data = _count_passthrough(data, skrub.var("load_counter"), "load")
    X = data["features"].skb.mark_as_X()
    y = data["target"].skb.mark_as_y()
    X = X.skb.apply(StandardScaler())
    X = _count_passthrough(X, skrub.var("scale_counter"), "scale")
    X = (
        X + X + X
    )  # without caching the _count_passthrough above would be computed 3 times per run
    pred = X.skb.apply(
        LogisticRegression(**skrub.choose_float(0.1, 1.0, name="C")), y=y
    )
    cv, search_iter, search_cv = 4, 3, 2
    search = pred.skb.get_randomized_search(n_iter=search_iter, cv=search_cv)
    data = {"load_counter": {}, "scale_counter": {}}
    skrub.cross_validate(search, data, cv=cv)
    assert data["load_counter"]["count"] == 1
    assert data["scale_counter"]["count"] == (
        # for each train-test split in outer loop
        cv
        * (
            # outer loop test
            1
            + (
                # refit best model
                1
                # for each hyper parameter
                + search_cv
                * (
                    # for each train-test split in inner loop
                    search_iter
                    * (
                        # train
                        1
                        # test
                        + 1
                    )
                )
            )
        )
    )


#
# misc utils from the _estimator module
#


def test_shared_dict():
    d = _SharedDict({"a": 0})
    assert clone(d, safe=False) is d
    assert copy.deepcopy(d) is d


#
# methods & attributes of the estimators
#


def test_get_params():
    e = (skrub.X() + skrub.choose_from([100, 200], name="a")).skb.apply(
        skrub.choose_from(
            {
                "dummy": DummyClassifier(),
                "logistic": LogisticRegression(
                    **skrub.choose_float(0.01, 1.0, log=True, name="C"),
                    **skrub.choose_from(["l1", "l2"], name="penalty"),
                ),
            },
            name="classifier",
        ),
        y=skrub.y(),
    )
    estimator = e.skb.get_estimator()
    params = {
        "expr",
        "expr__0",
        "expr__1",
        "expr__2",
        "expr__3",
    }
    assert estimator.get_params(deep=True).keys() == params
    assert estimator.get_params(deep=False).keys() == {"expr"}


def test_set_expr_in_params():
    e1 = skrub.var("a") + skrub.var("b")
    e2 = skrub.var("a") - skrub.var("b")
    estimator = e1.skb.get_estimator()
    data = {"a": 10, "b": 20}
    assert estimator.fit_transform(data) == 30
    estimator.set_params(expr=e2)
    assert estimator.fit_transform(data) == -10


def test_find_fitted_estimator():
    estimator = (
        (skrub.X() * 1.0)
        .skb.set_name("mul")
        .skb.apply(StandardScaler())
        .skb.set_name("scaler")
        .skb.apply(LogisticRegression(), y=skrub.y())
        .skb.set_name("predictor")
        .skb.get_estimator()
    )
    assert estimator.find_fitted_estimator("xyz") is None
    with pytest.raises(TypeError, match="Node 'X' does not represent"):
        estimator.find_fitted_estimator("X")
    with pytest.raises(ValueError, match="Node 'scaler' has not been fitted"):
        estimator.find_fitted_estimator("scaler")
    data = _simple_data()
    estimator.fit(data)
    assert isinstance(estimator.find_fitted_estimator("scaler"), StandardScaler)
    assert isinstance(estimator.find_fitted_estimator("predictor"), LogisticRegression)
