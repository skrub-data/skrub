import copy
import io
import pickle
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_classification
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.exceptions import NotFittedError
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils.validation import check_is_fitted

import skrub
from skrub._data_ops._estimator import _SharedDict
from skrub._data_ops._inspection import _has_graphviz

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


def get_data_op_and_data(data_kind):
    assert data_kind in ["simple", "unprocessed"]
    if data_kind == "simple":
        return _simple_data_classifier(), _simple_data()
    return _unprocessed_data_classifier(), _unprocessed_data()


@pytest.fixture(params=["simple", "unprocessed"])
def _data_op_and_data(request):
    data_kind = request.param
    data_op, data = get_data_op_and_data(data_kind)
    return {"data_kind": data_kind, "data_op": data_op, "data": data}


@pytest.fixture
def data_op(_data_op_and_data):
    return _data_op_and_data["data_op"]


@pytest.fixture
def data(_data_op_and_data):
    return _data_op_and_data["data"]


@pytest.fixture
def data_kind(_data_op_and_data):
    return _data_op_and_data["data_kind"]


@pytest.fixture(params=[None, 1, 2])
def n_jobs(request):
    return request.param


#
# fit, predict, param search & (possibly nested) cross-validation
#


def test_fit_predict():
    data_op, data = get_data_op_and_data("simple")
    learner = data_op.skb.make_learner()
    X_train, X_test, y_train, y_test = train_test_split(
        data["X"], data["y"], shuffle=False
    )
    with pytest.raises(NotFittedError):
        learner.predict({})
    learner.fit({"X": X_train, "y": y_train})

    # Note that y is missing from the environment here. It is not needed for
    # predict (or transform, score etc.)
    assert learner.decision_function({"X": X_test}).shape == (25,)
    predicted = learner.predict({"X": X_test})

    assert accuracy_score(y_test, predicted) == pytest.approx(0.84, abs=0.05)


def test_cross_validate(data_op, data, n_jobs):
    results = data_op.skb.cross_validate(data, n_jobs=n_jobs, return_learner=True)
    learners = results["learner"]
    assert len(learners) == 5
    for p in learners:
        assert p.__class__.__name__ == "SkrubLearner"
        assert is_fitted(p)
    score = results["test_score"]
    assert len(score) == 5

    assert score.mean() == pytest.approx(0.84, abs=0.05)


def test_return_estimator():
    data_op, data = get_data_op_and_data("simple")
    with pytest.raises(TypeError, match=".*return_learner"):
        data_op.skb.cross_validate(data, return_estimator=True)


def test_randomized_search(data_op, data, n_jobs):
    search = data_op.skb.make_randomized_search(n_iter=3, n_jobs=n_jobs, random_state=0)
    with pytest.raises(NotFittedError):
        search.predict(data)
    assert not hasattr(search, "results_")
    assert not hasattr(search, "detailed_results_")
    search.fit(data)
    assert list(search.results_.columns) == ["C", "mean_test_score"]
    assert list(search.detailed_results_.columns) == [
        "C",
        "std_score_time",
        "mean_score_time",
        "std_fit_time",
        "mean_fit_time",
        "std_test_score",
        "mean_test_score",
    ]
    assert search.results_["mean_test_score"].iloc[0] == pytest.approx(0.84, abs=0.05)
    assert search.decision_function(data).shape == (100,)
    train_score = search.score(data)
    assert train_score == pytest.approx(0.94)


def test_grid_search(data_op, data, n_jobs):
    search = data_op.skb.make_grid_search(n_jobs=n_jobs)
    search.fit(data)
    assert search.results_["mean_test_score"].iloc[0] == pytest.approx(0.84, abs=0.05)
    assert search.decision_function(data).shape == (100,)
    train_score = search.score(data)
    assert train_score == pytest.approx(0.94)


def test_no_names():
    X, y = make_classification(random_state=0, n_samples=20)
    e = (
        skrub.X(X)
        .skb.apply(StandardScaler(with_mean=skrub.choose_bool()))
        .skb.apply(StandardScaler(with_mean=skrub.choose_bool()))
        .skb.apply(StandardScaler(with_mean=skrub.choose_bool(name="m")))
        .skb.apply(LogisticRegression(), y=skrub.y(y))
    ).skb.make_grid_search(fitted=True, cv=2)
    assert list(e.results_.columns) == [
        "choose_bool()",
        "choose_bool()_1",
        "m",
        "mean_test_score",
    ]


def test_nested_cv(data_op, data, data_kind, n_jobs, monkeypatch):
    search = data_op.skb.make_randomized_search(n_iter=3, n_jobs=n_jobs, random_state=0)
    mock = Mock(side_effect=pd.read_csv)
    monkeypatch.setattr(pd, "read_csv", mock)

    results = skrub.cross_validate(search, data, n_jobs=n_jobs, return_learner=True)
    assert not is_fitted(search)
    score = results["test_score"]
    learners = results["learner"]
    assert len(learners) == 5
    for p in learners:
        assert is_fitted(p)
        assert p.__class__.__name__ == "ParamSearch"

    # when data is loaded from csv we check that the caching results in
    # read_csv being called exactly twice (we have 2 different 'files' to
    # load).
    assert mock.call_count == (2 if data_kind == "unprocessed" else 0)

    assert len(score) == 5
    assert score.mean() == pytest.approx(0.84, abs=0.05)


def test_unsupervised_no_y():
    X = np.random.default_rng(0).normal(size=(30, 20))
    data_op = skrub.X(X).skb.apply(
        PCA(**skrub.choose_from([4, 8], name="n_components"))
    )
    data_op_scores = skrub.cross_validate(
        data_op.skb.make_grid_search(), data_op.skb.get_data()
    )["test_score"]
    sklearn_search = GridSearchCV(PCA(), {"n_components": [4, 8]})
    sklearn_scores = cross_validate(sklearn_search, X)["test_score"]
    assert_allclose(sklearn_scores, data_op_scores)


def test_unsupervised():
    X, y = make_blobs(n_samples=10, random_state=0)
    k_means = KMeans(n_clusters=2, random_state=0, n_init=1)
    e = skrub.X(X).skb.apply(k_means, y=skrub.y(y), unsupervised=True)
    data_op_scores = e.skb.cross_validate()["test_score"]
    sklearn_scores = cross_validate(k_means, X, y)["test_score"]
    assert_allclose(sklearn_scores, data_op_scores)
    data_op_k_means = e.skb.make_learner()
    data_op_k_means.fit({"X": X})
    k_means.fit(X)
    assert (k_means.predict(X) == data_op_k_means.predict({"X": X})).all()
    assert_allclose(k_means.score(X, y), data_op_k_means.score({"X": X, "y": y}))
    with pytest.raises((KeyError, RuntimeError)):
        data_op_k_means.score({"X": X})


def test_no_apply_step():
    assert list(
        skrub.X().skb.cross_validate(
            {"X": np.ones((10, 2))}, cv=2, scoring=lambda e, X: 0
        )["test_score"]
    ) == [0, 0]


def test_multiclass():
    X, y = make_classification(n_classes=5, n_informative=10, random_state=0)
    data_op = skrub.X(X).skb.apply(
        LogisticRegression(**skrub.choose_from([0.001, 0.1], name="C"), random_state=0),
        y=skrub.y(y),
    )
    data_op_scores = skrub.cross_validate(
        data_op.skb.make_grid_search(), data_op.skb.get_data()
    )["test_score"]
    sklearn_search = GridSearchCV(
        LogisticRegression(random_state=0), {"C": [0.001, 0.1]}
    )
    sklearn_scores = cross_validate(sklearn_search, X, y)["test_score"]
    assert_allclose(sklearn_scores, data_op_scores)


def test_multimetric():
    X, y = make_classification(random_state=0)
    scoring = ["accuracy", "roc_auc"]
    data_op_search = (
        skrub.X(X)
        .skb.apply(
            LogisticRegression(**skrub.choose_from([0.001, 0.1], name="C")),
            y=skrub.y(y),
        )
        .skb.make_grid_search(fitted=True, scoring=scoring, refit="roc_auc")
    )
    assert list(data_op_search.results_.columns) == [
        "C",
        "mean_test_accuracy",
        "mean_test_roc_auc",
    ]
    assert list(data_op_search.detailed_results_.columns) == [
        "C",
        "std_score_time",
        "mean_score_time",
        "std_fit_time",
        "mean_fit_time",
        "std_test_accuracy",
        "std_test_roc_auc",
        "mean_test_accuracy",
        "mean_test_roc_auc",
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
        assert np.allclose(sklearn_results[col], data_op_search.results_[col].values)


def test_no_refit(data_op, data):
    search = data_op.skb.make_randomized_search(random_state=0, cv=2, refit=False).fit(
        data
    )
    assert search.best_params_["data_op__0"] == pytest.approx(0.01)
    assert search.results_.shape == (10, 2)
    with pytest.raises(
        AttributeError,
        match="This parameter search was initialized with `refit=False`",
    ):
        search.predict(data)


def test_multimetric_no_refit(data_op, data):
    search = data_op.skb.make_randomized_search(
        random_state=0, cv=2, refit=False, scoring=["accuracy", "roc_auc"]
    ).fit(data)
    assert not hasattr(search, "best_params_")
    assert search.results_.shape == (10, 3)


def test_when_last_step_is_not_apply(data_op, data):
    new_data_op = skrub.choose_from(
        [
            data_op.skb.apply_func(lambda x: x),
            data_op.skb.apply_func(lambda x: x),
        ],
        name="model",
    ).as_data_op()
    search = new_data_op.skb.make_randomized_search(
        n_iter=3,
        random_state=0,
    ).fit(data)
    assert search.results_.shape == (3, 3)
    assert search.results_["mean_test_score"].iloc[0] == pytest.approx(0.84, abs=0.05)
    assert search.decision_function(data).shape == (100,)
    train_score = search.score(data)
    assert train_score == pytest.approx(0.94)


@pytest.mark.parametrize("with_y", [False, True])
def test_train_test_split(with_y):
    n = skrub.var("n", 8)
    X = skrub.deferred(list)(skrub.deferred(range)(n)).skb.mark_as_X()
    y = skrub.deferred(list)(skrub.deferred(range)(n, 2 * n)).skb.mark_as_y()
    transformed = X[::-1]
    e = transformed + y if with_y else transformed
    split = e.skb.train_test_split(shuffle=False)
    if with_y:
        assert split == {
            "train": {
                "n": 8,
                "_skrub_X": [0, 1, 2, 3, 4, 5],
                "_skrub_y": [8, 9, 10, 11, 12, 13],
            },
            "test": {"n": 8, "_skrub_X": [6, 7], "_skrub_y": [14, 15]},
            "X_train": [0, 1, 2, 3, 4, 5],
            "X_test": [6, 7],
            "y_train": [8, 9, 10, 11, 12, 13],
            "y_test": [14, 15],
        }
    else:
        assert split == {
            "train": {"n": 8, "_skrub_X": [0, 1, 2, 3, 4, 5]},
            "test": {"n": 8, "_skrub_X": [6, 7]},
            "X_train": [0, 1, 2, 3, 4, 5],
            "X_test": [6, 7],
        }
    split = e.skb.train_test_split({"n": 4}, shuffle=False)
    if with_y:
        assert split == {
            "train": {"n": 4, "_skrub_X": [0, 1, 2], "_skrub_y": [4, 5, 6]},
            "test": {"n": 4, "_skrub_X": [3], "_skrub_y": [7]},
            "X_train": [0, 1, 2],
            "X_test": [3],
            "y_train": [4, 5, 6],
            "y_test": [7],
        }
        assert e.skb.eval(split["train"]) == [2, 1, 0, 4, 5, 6]
        assert e.skb.eval(split["test"]) == [3, 7]
        assert e.skb.eval() == [7, 6, 5, 4, 3, 2, 1, 0, 8, 9, 10, 11, 12, 13, 14, 15]
    else:
        assert split == {
            "train": {"n": 4, "_skrub_X": [0, 1, 2]},
            "test": {"n": 4, "_skrub_X": [3]},
            "X_train": [0, 1, 2],
            "X_test": [3],
        }
        assert e.skb.eval(split["train"]) == [2, 1, 0]
        assert e.skb.eval(split["test"]) == [3]
        assert e.skb.eval() == [7, 6, 5, 4, 3, 2, 1, 0]


def test_iter_cv_splits():
    X = skrub.X(np.arange(5) * 10)
    splits = X.skb.iter_cv_splits()
    s = next(splits)
    assert list(s["X_train"]) == list(s["train"]["_skrub_X"]) == [10, 20, 30, 40]
    assert list(s["X_test"]) == list(s["test"]["_skrub_X"]) == [0]
    s = next(splits)
    assert list(s["X_train"]) == list(s["train"]["_skrub_X"]) == [0, 20, 30, 40]
    assert list(s["X_test"]) == list(s["test"]["_skrub_X"]) == [10]

    X = skrub.X(np.arange(4) * 10)
    y = skrub.y(np.arange(4) * -10)
    splits = skrub.as_data_op((X, y)).skb.iter_cv_splits(cv=4)
    s = next(splits)
    assert list(s["X_train"]) == list(s["train"]["_skrub_X"]) == [10, 20, 30]
    assert list(s["X_test"]) == list(s["test"]["_skrub_X"]) == [0]
    assert list(s["y_train"]) == list(s["train"]["_skrub_y"]) == [-10, -20, -30]
    assert list(s["y_test"]) == list(s["test"]["_skrub_y"]) == [0]
    s = next(splits)
    assert list(s["X_train"]) == list(s["train"]["_skrub_X"]) == [0, 20, 30]
    assert list(s["X_test"]) == list(s["test"]["_skrub_X"]) == [10]
    assert list(s["y_train"]) == list(s["train"]["_skrub_y"]) == [0, -20, -30]
    assert list(s["y_test"]) == list(s["test"]["_skrub_y"]) == [-10]


def test_train_test_split_splitter_renaming():
    # TODO remove when `splitter` is removed in 0.7.0
    X = skrub.X(list(range(10)))

    def split(X, shuffle):
        return train_test_split(X, shuffle=shuffle)

    with pytest.warns(FutureWarning, match="`splitter`.*has been renamed"):
        assert X.skb.train_test_split(splitter=split, shuffle=False)["X_train"] == list(
            range(7)
        )


def test_iter_learners():
    e = skrub.choose_from([1, 2, 3], name="c").as_data_op()
    assert [p.describe_params() for p in e.skb.iter_learners_grid()] == [
        {"c": 1},
        {"c": 2},
        {"c": 3},
    ]

    e = skrub.choose_int(0, 1000, name="c").as_data_op()
    assert [
        p.describe_params() for p in e.skb.iter_learners_randomized(3, random_state=0)
    ] == [{"c": 549}, {"c": 715}, {"c": 603}]


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
    search = pred.skb.make_randomized_search(n_iter=search_iter, cv=search_cv)
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


@pytest.mark.parametrize(
    "e",
    [
        skrub.var("a").skb.apply_func(lambda x, f: x * f(), lambda: 2),
        skrub.as_data_op(lambda x: x * 2)(skrub.var("a")),
        (skrub.as_data_op([]) + [lambda x: x * 2])[0](skrub.var("a")),
    ],
)
def test_pickling(e):
    learner = pickle.loads(pickle.dumps(e.skb.make_learner()))
    assert learner.fit_transform({"a": 10}) == 20


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
    pytest.importorskip("plotly")
    data_op, data = get_data_op_and_data("simple")
    search = data_op.skb.make_randomized_search(n_iter=2, cv=2, random_state=0)
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


@pytest.mark.skipif(not _has_graphviz(), reason="full report requires graphviz")
def test_report(tmp_path):
    data_op, data = get_data_op_and_data("simple")
    pipe = data_op.skb.make_learner()
    with pytest.raises(NotFittedError):
        pipe.report(mode="score", environment=data)
    fit_report = pipe.report(
        mode="fit",
        environment=data,
        output_dir=tmp_path / "report",
        overwrite=True,
        open=False,
    )
    assert fit_report["result"] is pipe
    assert fit_report["error"] is None
    assert fit_report["report_path"].is_relative_to(tmp_path)
    score_report = pipe.report(
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
# methods & attributes of the learners
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
    learner = e.skb.make_learner()
    params = {
        "data_op",
        "data_op__0",
        "data_op__1",
        "data_op__2",
        "data_op__3",
    }
    assert learner.get_params(deep=True).keys() == params
    assert learner.get_params(deep=False).keys() == {"data_op"}


def test_set_data_op_in_params():
    e1 = skrub.var("a") + skrub.var("b")
    e2 = skrub.var("a") - skrub.var("b")
    learner = e1.skb.make_learner()
    data = {"a": 10, "b": 20}
    assert learner.fit_transform(data) == 30
    learner.set_params(data_op=e2)
    assert learner.fit_transform(data) == -10


def test_find_fitted_estimator():
    learner = (
        (skrub.X() * 1.0)
        .skb.set_name("mul")
        .skb.apply(StandardScaler())
        .skb.set_name("scaler")
        .skb.apply(LogisticRegression(), y=skrub.y())
        .skb.set_name("predictor")
        .skb.make_learner()
    )
    with pytest.raises(KeyError, match="'xyz'"):
        learner.find_fitted_estimator("xyz")
    with pytest.raises(TypeError, match="Node 'X' does not represent"):
        learner.find_fitted_estimator("X")
    with pytest.raises(ValueError, match="Node 'scaler' has not been fitted"):
        learner.find_fitted_estimator("scaler")
    data = _simple_data()
    learner.fit(data)
    assert isinstance(learner.find_fitted_estimator("scaler"), StandardScaler)
    assert isinstance(learner.find_fitted_estimator("predictor"), LogisticRegression)


def test_truncated_after():
    learner = (
        skrub.X()
        .skb.apply(MinMaxScaler())
        .skb.set_name("scaling")
        .skb.apply(LogisticRegression(), y=skrub.y())
        .skb.make_learner()
    )
    X = np.array([10.0, 5.0, 0.0])[:, None]
    y = np.array([1, 0, 1])
    learner.fit({"X": X, "y": y})
    sub_learner = learner.truncated_after("scaling")
    assert np.allclose(
        sub_learner.transform({"X": X}), np.array([1.0, 0.5, 0.0])[:, None]
    )
    X = np.array([100.0, 50.0, -10.0])[:, None]
    assert np.allclose(
        sub_learner.transform({"X": X}), np.array([10.0, 5.0, -1.0])[:, None]
    )
    with pytest.raises(KeyError, match="'xyz'"):
        learner.truncated_after("xyz")


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
        (SelectKBest, "transformer"),
    ],
)
@pytest.mark.parametrize("bury_apply", [False, True])
def test_estimator_type(estimator_type, expected, bury_apply):
    estimator = estimator_type()
    e = skrub.X().skb.apply(
        skrub.choose_from([estimator, estimator], name="_"), y=skrub.y()
    )
    if bury_apply:
        # add some choosing steps after the estimator so the root node is not
        # an Apply node
        e = (
            skrub.choose_from(["a", "b"], name="model")
            .match({"a": e, "b": e})
            .as_data_op()
        )
    for pipe in [
        e.skb.make_learner(),
        e.skb.make_grid_search(),
        e.skb.make_randomized_search(n_iter=2),
    ]:
        Xy_pipe = pipe.__skrub_to_Xy_pipeline__({})
        if hasattr(estimator, "_estimator_type"):
            assert Xy_pipe._estimator_type == expected
        if hasattr(estimator_type, "__sklearn_tags__"):
            # scikit-learn >= 1.6
            assert Xy_pipe.__sklearn_tags__() == estimator.__sklearn_tags__()
        else:
            assert not hasattr(Xy_pipe, "__sklearn_tags__")


def test_estimator_type_no_apply():
    e = skrub.X()
    for pipe in [
        e.skb.make_learner(),
        e.skb.make_grid_search(),
        e.skb.make_randomized_search(n_iter=2),
    ]:
        Xy_pipe = pipe.__skrub_to_Xy_pipeline__({})
        assert Xy_pipe._estimator_type == "transformer"
        if hasattr(BaseEstimator, "__sklearn_tags__"):
            # scikit-learn >= 1.6
            assert Xy_pipe.__sklearn_tags__().transformer_tags is not None
            assert Xy_pipe.__sklearn_tags__().classifier_tags is None
        else:
            assert not hasattr(Xy_pipe, "__sklearn_tags__")


@pytest.mark.parametrize("bury_apply", [False, True])
def test_classes(bury_apply):
    data_op, data = get_data_op_and_data("simple")
    if bury_apply:
        data_op = (
            skrub.choose_from(
                {"a": data_op.skb.apply_func(lambda x: x), "b": data_op},
                name="model",
            )
            .as_data_op()
            .skb.apply_func(lambda x: x)
        )
    logreg = LogisticRegression().fit(data["X"], data["y"])
    for pipe in [
        data_op.skb.make_learner(),
        data_op.skb.make_grid_search(),
        data_op.skb.make_randomized_search(n_iter=2),
    ]:
        Xy_pipe = pipe.__skrub_to_Xy_pipeline__({})
        assert not hasattr(Xy_pipe, "classes_")
        Xy_pipe.fit(data["X"], data["y"])
        assert (Xy_pipe.classes_ == logreg.classes_).all()


def test_classes_no_apply():
    data_op = skrub.X() + skrub.choose_from([0.0, 1.0], name="_")
    for pipe in [
        data_op.skb.make_learner(),
        data_op.skb.make_grid_search(scoring=lambda e, X: 0),
        data_op.skb.make_randomized_search(n_iter=2, scoring=lambda e, X: 0),
    ]:
        Xy_pipe = pipe.__skrub_to_Xy_pipeline__({})
        assert not hasattr(Xy_pipe, "classes_")
        Xy_pipe.fit(np.ones((10, 2)))
        assert not hasattr(Xy_pipe, "classes_")


@pytest.mark.parametrize("bury_apply", [False, True])
def test_support_modes(bury_apply):
    _, data = get_data_op_and_data("simple")
    choice = skrub.choose_from(["dummy", "logistic"], name="c")
    classif = choice.match(
        {"dummy": DummyClassifier(), "logistic": LogisticRegression()}
    )
    e = skrub.X().skb.apply(classif, y=skrub.y())
    if bury_apply:
        e = skrub.as_data_op({"a": e})["a"]
    learner = e.skb.make_learner()

    # as in grid-search, before fitting the learner's capabilities are read
    # from the default learner and after fitting from the fitted learner
    assert hasattr(learner, "predict")
    assert hasattr(learner, "predict_proba")
    assert not hasattr(learner, "decision_function")
    learner.fit(data | {"c": "logistic"})
    assert hasattr(learner, "decision_function")
    assert learner.decision_function(data).shape == data["y"].shape


def test_support_modes_no_apply():
    learner = skrub.X().skb.make_learner()
    for a in ["predict", "predict_proba", "score", "decision_function"]:
        assert not hasattr(learner, a)
    for a in ["fit", "transform", "fit_transform"]:
        assert hasattr(learner, a)


def test_random_search_no_vars():
    # non-regression test for #1600

    @skrub.deferred
    def load_data():
        X, y = make_classification(n_samples=10)
        return {"X": X, "y": y}

    data = load_data()
    X = data["X"].skb.mark_as_X()
    y = data["y"].skb.mark_as_y()
    pred = X.skb.apply(DummyClassifier(), y=y)
    search = pred.skb.make_grid_search(scoring="roc_auc").fit({})
    assert search.results_.shape[0] == 1
