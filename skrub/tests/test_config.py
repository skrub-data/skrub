import pytest

import skrub
from skrub import TableReport, config_context, get_config, set_config
from skrub._config import _parse_env_bool
from skrub._data_ops._evaluation import evaluate
from skrub.datasets import fetch_employee_salaries


def _use_table_report(obj):
    return "SkrubTableReport" in obj._repr_html_()


def test_config_context():
    assert get_config() == {
        "use_table_report": False,
        "use_table_report_data_ops": True,
        "max_plot_columns": 30,
        "max_association_columns": 30,
        "subsampling_seed": 0,
        "enable_subsampling": "default",
        "float_precision": 3,
        "cardinality_threshold": 40,
    }

    # Not using as a context manager affects nothing
    config_context(use_table_report=True)
    assert get_config()["use_table_report"] is False


def test_use_table_report_data_ops():
    X = skrub.X(fetch_employee_salaries().X)

    with config_context(use_table_report_data_ops=True):
        assert _use_table_report(X)
        with config_context(use_table_report_data_ops=False):
            assert not _use_table_report(X)


def test_use_table_report():
    X = fetch_employee_salaries().X
    assert not _use_table_report(X)
    with config_context(use_table_report=True):
        assert _use_table_report(X)
        with config_context(use_table_report=False):
            assert not _use_table_report(X)


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
        report = TableReport(X, max_association_columns="all", max_plot_columns="all")
        assert report.max_association_columns == "all"
        assert report.max_plot_columns == "all"

    # Check that max_plot_columns can be set after patching the TableReport
    # repr_html.
    with config_context(use_table_report=True):
        with config_context(max_plot_columns=3):
            "Plotting was skipped" in X._repr_html_()


def test_enable_subsampling():
    X = fetch_employee_salaries().X
    dataop = skrub.X(X)

    # No subsampling by default with fit_transform mode
    assert dataop.skb.subsample(n=3).skb.eval().shape[0] == X.shape[0]
    assert dataop.skb.subsample(n=3).skb.eval(keep_subsampling=True).shape[0] == 3

    # Force subsampling during fit_transform
    with config_context(enable_subsampling="force"):
        assert dataop.skb.subsample(n=3).skb.eval().shape[0] == 3

    # Default: subsampling during preview mode
    assert evaluate(dataop.skb.subsample(n=3), mode="preview").shape[0] == 3

    with config_context(enable_subsampling="disable"):
        assert (
            evaluate(dataop.skb.subsample(n=3), mode="preview").shape[0] == X.shape[0]
        )
        with config_context(enable_subsampling="default"):
            assert evaluate(dataop.skb.subsample(n=3), mode="preview").shape[0] == 3


def test_float_precision():
    y = fetch_employee_salaries().y

    # Default config: float_precision set to 3
    report = TableReport(y)
    mean = f"{report._summary['columns'][0]['mean']:#.3g}"
    html = report._repr_html_()
    assert mean in html

    # Float precision set to 2
    with config_context(float_precision=2):
        report_2 = TableReport(y)
        mean_2 = f"{report_2._summary['columns'][0]['mean']:#.2g}"
        html_2 = report_2._repr_html_()
        assert mean_2 in html_2

    assert mean != mean_2
    assert html != html_2


@pytest.mark.parametrize(
    "params",
    [
        {"use_table_report": "hello"},
        {"use_table_report_data_ops": 1},
        {"max_plot_columns": "hello"},
        {"max_association_columns": "hello"},
        {"subsampling_seed": -1},
        {"enable_subsampling": "no"},
        {"float_precision": -1},
        {"cardinality_threshold": -1},
    ],
)
def test_error(params):
    with pytest.raises(ValueError):
        set_config(**params)


def test_subsampling_seed():
    X = fetch_employee_salaries().X
    data_op = skrub.X(X)

    with config_context(subsampling_seed=0):
        index = evaluate(
            data_op.skb.subsample(n=3, how="random"), mode="preview"
        ).index.tolist()
        index_identical = evaluate(
            data_op.skb.subsample(n=3, how="random"), mode="preview"
        ).index.tolist()

    with config_context(subsampling_seed=1):
        index_different = evaluate(
            data_op.skb.subsample(n=3, how="random"), mode="preview"
        ).index.tolist()

    assert index == index_identical
    assert index != index_different


def test_parsing(monkeypatch):
    assert _parse_env_bool("MY_VAR", default=True)

    with monkeypatch.context() as m:
        m.setenv("MY_VAR", "False")
        assert not _parse_env_bool("MY_VAR", default=True)

    with monkeypatch.context() as m:
        m.setenv("MY_VAR", "True")
        assert _parse_env_bool("MY_VAR", default=False)

    with pytest.raises(ValueError):
        with monkeypatch.context() as m:
            m.setenv("MY_VAR", "hello")
            _parse_env_bool("MY_VAR", default=False)
