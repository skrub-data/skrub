import pytest

import skrub
from skrub import TableReport, config_context, get_config, set_config
from skrub._expressions._evaluation import evaluate
from skrub.datasets import fetch_employee_salaries


def _use_tablereport(obj):
    return "SkrubTableReport" in obj._repr_html_()


def test_config_context():
    assert get_config() == {
        "use_tablereport": False,
        "use_tablereport_expr": True,
        "max_plot_columns": 30,
        "max_association_columns": 30,
        "subsampling_seed": 0,
        "enable_subsampling": "default",
    }

    # Not using as a context manager affects nothing
    config_context(use_tablereport=True)
    assert get_config()["use_tablereport"] is False


def test_use_tablereport_expr():
    X = skrub.X(fetch_employee_salaries().X)

    with config_context(use_tablereport_expr=True):
        assert _use_tablereport(X)
        with config_context(use_tablereport_expr=False):
            assert not _use_tablereport(X)


def test_use_tablereport():
    X = fetch_employee_salaries().X
    assert not _use_tablereport(X)
    with config_context(use_tablereport=True):
        assert _use_tablereport(X)
        with config_context(use_tablereport=False):
            assert not _use_tablereport(X)


def test_max_plot_columns():
    X = fetch_employee_salaries().X
    report = TableReport(X)
    assert report.max_association_columns == 30
    assert report.max_plot_columns == 30

    # Set default to 1
    with config_context(max_plot_columns=1):
        report = TableReport(X)
        assert report.max_association_columns == 30
        assert report.max_plot_columns == 1

        # Argument takes precedence over default configuration
        report = TableReport(X, max_association_columns=12)
        assert report.max_association_columns == 12
        assert report.max_plot_columns == 1

    # Check that max_plot_columns can be set after patching the TableReport
    # repr_html.
    with config_context(use_tablereport=True):
        with config_context(max_plot_columns=3):
            "Plotting was skipped" in X._repr_html_()


def test_enable_subsampling():
    X = fetch_employee_salaries().X
    expr = skrub.X(X)

    # Default: no subsampling during fit mode
    assert expr.skb.subsample(n=3).skb.eval().shape[0] == X.shape[0]

    # Force subsampling
    with config_context(enable_subsampling="force"):
        assert expr.skb.subsample(n=3).skb.eval().shape[0] == 3

    # Default: subsampling during preview mode
    assert evaluate(expr.skb.subsample(n=3), mode="preview").shape[0] == 3

    with config_context(enable_subsampling="disable"):
        assert evaluate(expr.skb.subsample(n=3), mode="preview").shape[0] == X.shape[0]
        with config_context(enable_subsampling="default"):
            assert evaluate(expr.skb.subsample(n=3), mode="preview").shape[0] == 3


@pytest.mark.parametrize(
    "params",
    [
        {"use_tablereport": "hello"},
        {"use_tablereport_expr": 1},
        {"max_plot_columns": "hello"},
        {"max_association_columns": "hello"},
        {"subsampling_seed": -1},
        {"enable_subsampling": "no"},
    ],
)
def test_error(params):
    with pytest.raises(ValueError):
        set_config(**params)


def test_subsampling_seed():
    X = fetch_employee_salaries().X
    expr = skrub.X(X)

    with config_context(subsampling_seed=0):
        index = evaluate(
            expr.skb.subsample(n=3, how="random"), mode="preview"
        ).index.tolist()
        index_identical = evaluate(
            expr.skb.subsample(n=3, how="random"), mode="preview"
        ).index.tolist()

    with config_context(subsampling_seed=1):
        index_different = evaluate(
            expr.skb.subsample(n=3, how="random"), mode="preview"
        ).index.tolist()

    assert index == index_identical
    assert index != index_different
