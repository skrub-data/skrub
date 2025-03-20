import copy
import io
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

import skrub
from skrub._expressions._estimator import _SharedDict

#
# testing utils
#


def is_fitted(estimator):
    try:
        check_is_fitted(estimator)
        return True
    except NotFittedError:
        return False


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
    with pytest.raises(NotFittedError):
        estimator.predict({})
    estimator.fit({"X": X_train, "y": y_train})

    # Note that y is missing from the environment here. It is not needed for
    # predict (or transform, score etc.)
    assert estimator.decision_function({"X": X_test}).shape == (25,)
    predicted = estimator.predict({"X": X_test})

    assert accuracy_score(y_test, predicted) == pytest.approx(0.84, abs=0.05)


def test_cross_validate(expression, data, n_jobs):
    results = expression.skb.cross_validate(data, n_jobs=n_jobs, return_estimator=True)
    estimators = results["estimator"]
    assert len(estimators) == 5
    for e in estimators:
        assert e.__class__.__name__ == "ExprEstimator"
        assert is_fitted(e)
    score = results["test_score"]
    assert len(score) == 5

    assert score.mean() == pytest.approx(0.84, abs=0.05)


def test_randomized_search(expression, data, n_jobs):
    search = expression.skb.get_randomized_search(
        n_iter=3, n_jobs=n_jobs, random_state=0
    )
    with pytest.raises(NotFittedError):
        search.predict(data)
    assert not hasattr(search, "results_")
    assert not hasattr(search, "detailed_results_")
    search.fit(data)
    assert list(search.results_.columns) == ["mean_test_score", "C"]
    assert list(search.detailed_results_.columns) == [
        "mean_test_score",
        "C",
        "std_test_score",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    ]
    assert search.results_["mean_test_score"].iloc[0] == pytest.approx(0.84, abs=0.05)
    assert search.decision_function(data).shape == (100,)
    train_score = search.score(data)
    assert train_score == pytest.approx(0.94)


def test_grid_search(expression, data, n_jobs):
    search = expression.skb.get_grid_search(n_jobs=n_jobs)
    search.fit(data)
    search.results_["mean_test_score"].iloc[0] == pytest.approx(0.84, abs=0.05)
    assert search.decision_function(data).shape == (100,)
    train_score = search.score(data)
    assert train_score == pytest.approx(0.94)


def test_nested_cv(expression, data, data_kind, n_jobs, monkeypatch):
    search = expression.skb.get_randomized_search(
        n_iter=3, n_jobs=n_jobs, random_state=0
    )
    mock = Mock(side_effect=pd.read_csv)
    monkeypatch.setattr(pd, "read_csv", mock)

    results = skrub.cross_validate(search, data, n_jobs=n_jobs, return_estimator=True)
    assert not is_fitted(search)
    score = results["test_score"]
    estimators = results["estimator"]
    assert len(estimators) == 5
    for e in estimators:
        assert is_fitted(e)
        assert e.__class__.__name__ == "ParamSearch"

    # when data is loaded from csv we check that the caching results in
    # read_csv being called exactly twice (we have 2 different 'files' to
    # load).
    assert mock.call_count == (2 if data_kind == "unprocessed" else 0)

    assert len(score) == 5
    assert score.mean() == pytest.approx(0.84, abs=0.05)


def test_unsupervised():
    X = np.random.default_rng(0).normal(size=(30, 20))
    expr = skrub.X(X).skb.apply(PCA(**skrub.choose_from([4, 8], name="n_components")))
    expr_scores = skrub.cross_validate(expr.skb.get_grid_search(), expr.skb.get_data())[
        "test_score"
    ]
    sklearn_search = GridSearchCV(PCA(), {"n_components": [4, 8]})
    sklearn_scores = cross_validate(sklearn_search, X)["test_score"]
    assert_allclose(sklearn_scores, expr_scores)


def test_multiclass():
    X, y = make_classification(n_classes=5, n_informative=10, random_state=0)
    expr = skrub.X(X).skb.apply(
        LogisticRegression(**skrub.choose_from([0.001, 0.1], name="C"), random_state=0),
        y=skrub.y(y),
    )
    expr_scores = skrub.cross_validate(expr.skb.get_grid_search(), expr.skb.get_data())[
        "test_score"
    ]
    sklearn_search = GridSearchCV(
        LogisticRegression(random_state=0), {"C": [0.001, 0.1]}
    )
    sklearn_scores = cross_validate(sklearn_search, X, y)["test_score"]
    assert_allclose(sklearn_scores, expr_scores)


def test_multimetric():
    X, y = make_classification(random_state=0)
    scoring = ["accuracy", "roc_auc"]
    expr_search = (
        skrub.X(X)
        .skb.apply(
            LogisticRegression(**skrub.choose_from([0.001, 0.1], name="C")),
            y=skrub.y(y),
        )
        .skb.get_grid_search(fitted=True, scoring=scoring, refit="roc_auc")
    )
    assert list(expr_search.results_.columns) == [
        "mean_test_roc_auc",
        "mean_test_accuracy",
        "C",
    ]
    assert list(expr_search.detailed_results_.columns) == [
        "mean_test_roc_auc",
        "mean_test_accuracy",
        "C",
        "std_test_roc_auc",
        "std_test_accuracy",
        "mean_fit_time",
        "std_fit_time",
        "mean_score_time",
        "std_score_time",
    ]

    sklearn_results = (
        GridSearchCV(
            LogisticRegression(), {"C": [0.001, 0.1]}, scoring=scoring, refit="roc_auc"
        )
        .fit(X, y)
        .cv_results_
    )
    for metric in scoring:
        col = f"mean_test_{metric}"
        assert np.allclose(sklearn_results[col], expr_search.results_[col].values)


def test_no_refit(expression, data):
    search = expression.skb.get_randomized_search(
        random_state=0, cv=2, refit=False
    ).fit(data)
    assert search.best_params_["expr__0"] == pytest.approx(0.01)
    assert search.results_.shape == (10, 2)
    with pytest.raises(
        AttributeError,
        match="This parameter search was initialized with `refit=False`",
    ):
        search.predict(data)


#
# caching
#


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
# reporting & plotting
#


def test_plot_results():
    expr, data = get_expression_and_data("simple")
    search = expr.skb.get_randomized_search(n_iter=2, cv=2, random_state=0)
    with pytest.raises(NotFittedError):
        search.plot_results()
    search.fit(data)
    with pytest.raises(ValueError, match="No results to plot"):
        search.plot_results(min_score=2.0)
    for min_score in [None, 0.1]:
        try:
            import plotly  # noqa: F401

            plotly_installed = True
        except ImportError:
            plotly_installed = False
        fig = search.plot_results(min_score=min_score)
        assert (fig is None) == (not plotly_installed)


def test_report(tmp_path):
    expr, data = get_expression_and_data("simple")
    est = expr.skb.get_estimator()
    with pytest.raises(NotFittedError):
        est.report(mode="score", environment=data)
    fit_report = est.report(
        mode="fit",
        environment=data,
        output_dir=tmp_path / "report",
        overwrite=True,
        open=False,
    )
    assert isinstance(fit_report["result"], LogisticRegression)
    assert fit_report["error"] is None
    assert fit_report["report_path"].is_relative_to(tmp_path)
    score_report = est.report(
        mode="score",
        environment=data,
        output_dir=tmp_path / "report",
        overwrite=True,
        open=False,
    )
    assert score_report["result"] == pytest.approx(0.94, abs=0.1)
    assert score_report["error"] is None
    assert score_report["report_path"].is_relative_to(tmp_path)


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


#
# methods of private types for compatibility with GridSearchCV, cross_validate etc.
#


# In old scikit-learn versions it uses attributes like _estimator_type, in
# recent versions the __sklearn_tags__


@pytest.mark.parametrize(
    "estimator_type, expected",
    [
        (LogisticRegression, "classifier"),
        (DummyClassifier, "classifier"),
        (Ridge, "regressor"),
        (DummyRegressor, "regressor"),
        (SelectKBest, "transformer"),
    ],
)
def test_estimator_type(estimator_type, expected):
    estimator = estimator_type()
    e = skrub.X().skb.apply(
        skrub.choose_from([estimator, estimator], name="_"), y=skrub.y()
    )
    for est in [
        e.skb.get_estimator(),
        e.skb.get_grid_search(),
        e.skb.get_randomized_search(n_iter=2),
    ]:
        Xy_est = est.__skrub_to_Xy_estimator__({})
        assert Xy_est._estimator_type == expected
        if hasattr(estimator_type, "__sklearn_tags__"):
            # scikit-learn >= 1.6
            assert Xy_est.__sklearn_tags__() == estimator.__sklearn_tags__()
        else:
            assert not hasattr(Xy_est, "__sklearn_tags__")


def test_classes():
    expression, data = get_expression_and_data("simple")
    logreg = LogisticRegression().fit(data["X"], data["y"])
    for est in [
        expression.skb.get_estimator(),
        expression.skb.get_grid_search(),
        expression.skb.get_randomized_search(n_iter=2),
    ]:
        Xy_est = est.__skrub_to_Xy_estimator__({})
        assert not hasattr(Xy_est, "classes_")
        Xy_est.fit(data["X"], data["y"])
        assert (Xy_est.classes_ == logreg.classes_).all()
