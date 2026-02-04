import pathlib

import pytest

import skrub
from skrub import TableReport, config_context, get_config, set_config
from skrub._config import _parse_env_bool
from skrub._data_ops._evaluation import evaluate
from skrub.conftest import skip_polars_installed_without_pyarrow


def _use_table_report(obj):
    return "SkrubTableReport" in obj._repr_html_()


@pytest.fixture
def simple_df(df_module):
    return df_module.make_dataframe(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "a", "b", "c"],
        }
    )


@pytest.fixture
def simple_series(df_module):
    return df_module.make_column(name="A", values=[1, 2, 3, 4, 5])


def test_default_config():
    cfg = get_config()
    # Rather than asserting that the dictionary is exactly equal to the hard-coded
    # default values, we check each expected default value individually.
    assert cfg["use_table_report"] is False
    assert cfg["use_table_report_data_ops"] is True
    # On CI the absolute path is different, check that it ends with skrub_data
    assert pathlib.Path(cfg["data_folder"]).name == "skrub_data"
    assert cfg["table_report_verbosity"] == 1
    assert cfg["max_plot_columns"] == 30
    assert cfg["max_association_columns"] == 30
    assert cfg["subsampling_seed"] == 0
    assert cfg["enable_subsampling"] == "default"
    assert cfg["float_precision"] == 3
    assert cfg["cardinality_threshold"] == 40

    # Fail the test if new configuration keys are present but not checked here.
    expected_keys = {
        "use_table_report",
        "use_table_report_data_ops",
        "data_folder",
        "table_report_verbosity",
        "max_plot_columns",
        "max_association_columns",
        "subsampling_seed",
        "enable_subsampling",
        "float_precision",
        "cardinality_threshold",
    }
    assert set(cfg.keys()) == expected_keys


def test_config_context():
    # Default value
    assert get_config()["use_table_report"] is False
    # Not using as a context manager affects nothing
    config_context(use_table_report=True)
    assert get_config()["use_table_report"] is False


def test_use_table_report_data_ops(simple_df):
    X = skrub.X(simple_df)
    with config_context(use_table_report_data_ops=True):
        assert _use_table_report(X)
        with config_context(use_table_report_data_ops=False):
            assert not _use_table_report(X)


@skip_polars_installed_without_pyarrow
def test_use_table_report(simple_df):
    assert not _use_table_report(simple_df)
    with config_context(use_table_report=True):
        assert _use_table_report(simple_df)
        with config_context(use_table_report=False):
            assert not _use_table_report(simple_df)


@skip_polars_installed_without_pyarrow
def test_max_plot_columns(simple_df):
    report = TableReport(simple_df)
    assert report.max_association_columns == 30
    assert report.max_plot_columns == 30

    # Set default to 1
    with config_context(max_plot_columns=1):
        report = TableReport(simple_df)
        assert report.max_association_columns == 30
        assert report.max_plot_columns == 1

        # Argument takes precedence over default configuration
        report = TableReport(
            simple_df, max_association_columns="all", max_plot_columns="all"
        )
        assert report.max_association_columns == "all"
        assert report.max_plot_columns == "all"

    # Check that max_plot_columns can be set after patching the TableReport
    # repr_html.
    with config_context(use_table_report=True):
        with config_context(max_plot_columns=3):
            "Plotting was skipped" in simple_df._repr_html_()


def test_enable_subsampling(simple_df):
    dataop = skrub.X(simple_df)

    # No subsampling by default with fit_transform mode
    assert dataop.skb.subsample(n=3).skb.eval().shape[0] == simple_df.shape[0]
    assert dataop.skb.subsample(n=3).skb.eval(keep_subsampling=True).shape[0] == 3

    # Force subsampling during fit_transform
    with config_context(enable_subsampling="force"):
        assert dataop.skb.subsample(n=3).skb.eval().shape[0] == 3

    # Default: subsampling during preview mode
    assert evaluate(dataop.skb.subsample(n=3), mode="preview").shape[0] == 3

    with config_context(enable_subsampling="disable"):
        assert (
            evaluate(dataop.skb.subsample(n=3), mode="preview").shape[0]
            == simple_df.shape[0]
        )
        with config_context(enable_subsampling="default"):
            assert evaluate(dataop.skb.subsample(n=3), mode="preview").shape[0] == 3


@skip_polars_installed_without_pyarrow
def test_float_precision(simple_series):
    # Default config: float_precision set to 3
    report = TableReport(simple_series)
    mean = f"{report._summary['columns'][0]['mean']:#.3g}"
    html = report._repr_html_()
    assert mean in html

    # Float precision set to 2
    with config_context(float_precision=2):
        report_2 = TableReport(simple_series)
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


def test_subsampling_seed(simple_df):
    data_op = skrub.X(simple_df)

    with config_context(subsampling_seed=0):
        col_a = evaluate(data_op.skb.subsample(n=3, how="random"), mode="preview")[
            "A"
        ].to_list()
        col_a_identical = evaluate(
            data_op.skb.subsample(n=3, how="random"), mode="preview"
        )["A"].to_list()

    with config_context(subsampling_seed=1):
        col_a_different = evaluate(
            data_op.skb.subsample(n=3, how="random"), mode="preview"
        )["A"].to_list()

    assert col_a == col_a_identical
    assert col_a != col_a_different


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


def test_wrong_verbosity():
    with pytest.raises(ValueError, match=".*table_report_verbosity.*"):
        with config_context(table_report_verbosity=-1):
            pass
