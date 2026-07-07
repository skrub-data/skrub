import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier

import skrub
from skrub._data_ops._parallel_coord import _add_jitter, _prepare_column


def _has_plotly():
    try:
        import plotly  # noqa: F401

        return True
    except ImportError:
        return False


def test_no_plotly():
    if _has_plotly():
        return

    X_a, y_a = make_classification(n_samples=20, n_features=4, n_informative=2)
    X, y = skrub.X(X_a), skrub.y(y_a)
    pred = X.skb.apply(
        DummyClassifier(
            **skrub.choose_from(["most_frequent", "prior"], name="strategy")
        ),
        y=y,
    )
    search = pred.skb.make_randomized_search(fitted=True)
    assert search.results_.shape == (2, 2)
    with pytest.raises(ImportError, match="Please install plotly"):
        search.plot_results()


def test_parallel_coord():
    X_a, y_a = make_classification(n_samples=20, n_features=4, n_informative=2)
    c0 = skrub.choose_from({"a": 0, "b": 1}, name="c0")
    c1 = skrub.choose_from([0, 1], name="c1")

    c2 = skrub.choose_int(1, 100, log=True, name="c2")
    c3 = skrub.choose_float(0.0, 1.0, name="c3")
    c4 = skrub.choose_from({"A": 101, "z": 102}, name="c4")
    c5 = skrub.choose_bool(name="c5")
    c6 = skrub.choose_from([2, 3, 4, 5], name="c6").match({2: c2, 3: c3, 4: c4, 5: c5})
    c7 = skrub.choose_int(1, 100, log=True, name="c7")
    c9 = skrub.choose_from([skrub.choose_int(1, 3, name="c8"), 4], name="c9")

    X = skrub.as_data_op([skrub.X(), c0, c1, c6, c7, c9])[0]
    pred = X.skb.apply(DummyClassifier(), y=skrub.y())
    search = pred.skb.make_randomized_search(random_state=0, n_iter=30).fit(
        {"X": X_a, "y": y_a}
    )

    pytest.importorskip("plotly")

    fig = search.plot_results()
    data = iter(fig.data[0]["dimensions"])
    dim = next(data)
    assert dim["label"] == "c0"
    assert list(dim["ticktext"]) == ["a", "b"]
    assert list(dim["tickvals"]) == [0, 1]
    next(data)
    next(data)
    next(data)
    next(data)
    dim = next(data)
    assert dim["label"] == "c4"
    assert list(dim["ticktext"]) == ["Null", "A", "z"]
    assert list(dim["tickvals"]) == [-1.0, 0, 1]
    dim = next(data)
    assert dim["label"] == "c5"
    assert list(dim["ticktext"]) == ["Null", "False", "True"]
    assert list(dim["tickvals"]) == [-1.0, 0, 1]
    next(data)
    dim = next(data)
    assert dim["label"] == "c8"
    assert list(dim["ticktext"]) == ["NaN", "1", "2", "3"]
    dim = next(data)
    assert dim["label"] == "c9"
    assert list(dim["ticktext"]) == ["4", "choose_int(1, 3, name='c8')"]
    dim = next(data)
    assert dim["label"] == "score time"
    dim = next(data)
    assert dim["label"] == "fit time"
    dim = next(data)
    assert dim["label"] == "score"


@pytest.mark.parametrize(
    "is_log_scale, is_int",
    [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ],
)
def test_column_preparation(is_log_scale, is_int):
    pytest.importorskip("plotly")

    # Normal case
    column = pd.Series(np.array([1.0, 2.0, 3.0]), name="test")
    prepared = _prepare_column(column, is_log_scale=is_log_scale, is_int=is_int)

    if is_log_scale:
        expected_values = np.log(np.array([1.0, 2.0, 3.0]))
    else:
        expected_values = np.array([1.0, 2.0, 3.0])

    assert np.allclose(prepared["values"], expected_values)

    # One non-nan value
    column = pd.Series(np.array([np.nan, 2.0, np.nan]), name="test")
    prepared = _prepare_column(column, is_log_scale=is_log_scale, is_int=is_int)

    if is_log_scale:
        expected_value = np.log(2.0)
    else:
        expected_value = 2.0

    # check that the non-nan value is correctly placed
    assert np.allclose(
        prepared["values"],
        [prepared["tickvals"][0], expected_value, prepared["tickvals"][0]],
    )
    assert prepared["ticktext"] == ["NaN", "2"]

    # All nans
    column = pd.Series(np.array([np.nan, np.nan, np.nan]), name="test")

    prepared = _prepare_column(column, is_log_scale=is_log_scale, is_int=is_int)
    jittered = _add_jitter(prepared)

    assert prepared["ticktext"] == ["NaN"]
    assert np.all(prepared["values"] == 0.0)
    assert np.all(jittered["values"] == 0.0)

    # All identical values
    column = pd.Series(np.array([1.0, 1.0, 1.0]), name="test")

    prepared = _prepare_column(column, is_log_scale=is_log_scale, is_int=is_int)
    jittered = _add_jitter(prepared)

    assert prepared["ticktext"] == ["1"]
    if is_log_scale:
        assert np.all(prepared["values"] == np.log(1.0))
        assert np.all(jittered["values"] == np.log(1.0))
    else:
        assert np.all(prepared["values"] == 1.0)
        assert np.all(jittered["values"] == 1.0)


@pytest.fixture(scope="module", params=[False, True])
def classif_grid_search(request):
    use_with_scoring = request.param
    pytest.importorskip("plotly")

    X, y = make_classification()
    X = pd.DataFrame(X)
    X.columns = [str(c) for c in X.columns]
    X, y = skrub.X(X), skrub.y(y)

    cols = skrub.choose_from([["0"], ["1"]], name="cols")
    add = skrub.choose_float(0.0, 1.0, name="add", n_steps=3)
    mul = skrub.choose_float(1.0, 2.0, n_steps=3)  # no name
    pred = ((X[cols] + add) * mul).skb.apply(DummyClassifier(), y=y)
    if use_with_scoring:
        return pred.skb.with_scoring(
            ["accuracy", "neg_brier_score"]
        ).skb.make_grid_search(
            fitted=True,
            refit="accuracy",
        )
    else:
        return pred.skb.make_grid_search(
            fitted=True,
            scoring=["accuracy", "neg_brier_score"],
            refit="accuracy",
        )


@pytest.mark.parametrize(
    "show_scores, show_choices, show_times, expected",
    [
        (
            None,
            None,
            None,
            [
                "cols",
                "add",
                "choose_float(1.0,<br>\n2.0, n_steps=3)",
                "score time",
                "fit time",
                "mean_test_neg_brier_<br>\nscore",
                "mean_test_accuracy",
            ],
        ),
        (
            "accuracy",
            ["cols", "add"],
            "fit",
            [
                "cols",
                "add",
                "fit time",
                "mean_test_accuracy",
            ],
        ),
        (
            [],
            "add",
            [],
            [
                "add",
            ],
        ),
    ],
)
def test_multi_scoring_and_filtering(
    classif_grid_search, show_scores, show_choices, show_times, expected
):
    fig = classif_grid_search.plot_results(
        show_scores=show_scores, show_choices=show_choices, show_times=show_times
    )
    dimensions = fig.data[0]["dimensions"]
    assert [d["label"] for d in dimensions] == expected


def test_bad_filtering_params(classif_grid_search):
    # Ask for a score that is not available, available scores are shown
    with pytest.raises(ValueError, match="['accuracy', 'neg_brier_score']"):
        classif_grid_search.plot_results(show_scores=["roc_auc"])
    # Only choices with an actual name can be selected. Here we ask for a
    # choice name that does not exist, available ones are shown.
    with pytest.raises(ValueError, match="['cols', 'add']"):
        classif_grid_search.plot_results(
            show_choices=["choose_float(1.0, 2.0, n_steps=3)"]
        )
