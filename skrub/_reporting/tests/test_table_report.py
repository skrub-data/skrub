import contextlib
import datetime
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.utils import Bunch

from skrub import TableReport, ToDatetime
from skrub import _dataframe as sbd
from skrub._reporting._sample_table import make_table
from skrub.conftest import skip_polars_installed_without_pyarrow


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


def get_report_id(html):
    return re.search(r'<skrub-table-report.*?id="report_([a-z0-9]+)"', html).group(1)


@skip_polars_installed_without_pyarrow
def test_report(air_quality):
    col_filt = {
        "first_2": {
            "display_name": "First 2",
            "columns": sbd.column_names(air_quality)[:2],
        }
    }
    report = TableReport(air_quality, title="the title", column_filters=col_filt)
    assert report.max_association_columns == 30
    html = report.html()
    assert "the title" in html
    assert "With nulls" in html
    assert "First 10" in html
    assert "First 2" in html
    for col_name in sbd.column_names(air_quality):
        assert col_name in html
    report_id = get_report_id(html)
    assert len(report_id) == 8
    all_report_ids = [report_id]
    report_id = get_report_id(report.html())
    assert len(report_id) == 8
    all_report_ids.append(report_id)
    snippet = report.html_snippet()
    report_id = get_report_id(snippet)
    all_report_ids.append(report_id)
    assert "<html" not in snippet
    assert "the title" in snippet
    assert "<skrub-table-report" in snippet
    data = json.loads(report.json())
    assert data["title"] == "the title"
    assert report._summary["title"] == "the title"
    assert report._summary["title"] == "the title"
    snippet = report._repr_mimebundle_()["text/html"]
    report_id = get_report_id(snippet)
    all_report_ids.append(report_id)
    assert "<html" not in snippet
    assert "the title" in snippet
    assert "<skrub-table-report" in snippet
    snippet = report._repr_html_()
    report_id = get_report_id(snippet)
    all_report_ids.append(report_id)
    assert "<html" not in snippet
    assert "the title" in snippet
    assert "<skrub-table-report" in snippet
    assert len(all_report_ids) == len(set(all_report_ids))


@skip_polars_installed_without_pyarrow
def test_few_columns(df_module, check_polars_numpy2):
    report = TableReport(df_module.example_dataframe)
    assert "First 10 columns" not in report.html()


@skip_polars_installed_without_pyarrow
def test_few_rows(df_module, check_polars_numpy2):
    df = sbd.slice(df_module.example_dataframe, 2)
    TableReport(df).html()


def test_empty_dataframe(df_module):
    html = TableReport(df_module.empty_dataframe).html()
    assert "The dataframe is empty." in html


def test_open(pd_module, browser_mock):
    TableReport(pd_module.example_dataframe, title="the title").open()
    assert b"the title" in browser_mock.content


@skip_polars_installed_without_pyarrow
def test_non_hashable_values(df_module):
    # non-regression test for #1066
    df = df_module.make_dataframe(dict(a=[[1, 2, 3], None, [4]]))
    html = TableReport(df).html()
    assert "[1, 2, 3]" in html


@skip_polars_installed_without_pyarrow
def test_nat(df_module):
    # non-regression for:
    # https://github.com/skrub-data/skrub/issues/1111
    # NaT used to cause exception when plotting histogram
    col = df_module.make_column(
        "a", ["2020-01-01T01:00:00 UTC", "2020-01-02T01:00:00 UTC", None]
    )
    col = ToDatetime().fit_transform(col)
    df = df_module.make_dataframe({"a": col})
    TableReport(df).html()


@skip_polars_installed_without_pyarrow
def test_bool_column_mean(df_module):
    df = df_module.make_dataframe({"a": [True, False, True, True, False, True]})
    html = TableReport(df).html()
    assert "Mean" in html
    assert "0.667" in html


def test_duplicate_columns(pd_module):
    df = pd_module.make_dataframe({"a": [1, 2], "b": [3, 4]})
    df.columns = ["a", "a"]
    TableReport(df).html()


@skip_polars_installed_without_pyarrow
def test_infinite_values(df_module):
    # Non-regression for https://github.com/skrub-data/skrub/issues/1134
    # (histogram plot failing with infinite values)
    with warnings.catch_warnings():
        # convert_dtypes() emits spurious warning while deciding whether to cast to int
        warnings.filterwarnings("ignore", message="invalid value encountered in cast")
        df = df_module.make_dataframe(
            dict(a=[float("inf"), 1.0, 2.0], b=[0.0, 1.0, 2.0])
        )

    TableReport(df).html()


@skip_polars_installed_without_pyarrow
def test_duration(df_module):
    df = df_module.make_dataframe(
        {"a": [datetime.timedelta(days=2), datetime.timedelta(days=3)]}
    )
    assert re.search(r"2(\.0)?\s+days", TableReport(df).html())


@pytest.mark.parametrize(
    "filename_type",
    ["str", "Path", "text_file_object", "binary_file_object"],
)
def test_write_html(tmp_path, pd_module, filename_type):
    df = pd_module.make_dataframe({"a": [1, 2], "b": [3, 4]})
    report = TableReport(df)

    tmp_file_path = tmp_path / Path("report.html")

    # making sure we are closing the open files, and dealing with the first
    # condition which doesn't require opening any file
    with contextlib.ExitStack() as stack:
        if filename_type == "str":
            filename = str(tmp_file_path)
        elif filename_type == "text_file_object":
            filename = stack.enter_context(open(tmp_file_path, "w", encoding="utf-8"))
        elif filename_type == "binary_file_object":
            filename = stack.enter_context(open(tmp_file_path, "wb"))
        else:
            filename = tmp_file_path

        report.write_html(filename)
        assert tmp_file_path.exists()

    with open(tmp_file_path, "r", encoding="utf-8") as file:
        saved_content = file.read()
    assert "</html>" in saved_content


def test_write_html_with_not_utf8_encoding(tmp_path, pd_module):
    df = pd_module.make_dataframe({"a": [1, 2], "b": [3, 4]})
    report = TableReport(df)
    tmp_file_path = tmp_path / Path("report.html")

    with open(tmp_file_path, "w", encoding="latin-1") as file:
        encoding = getattr(file, "encoding", None)
        with pytest.raises(
            ValueError,
            match=(
                "If `file` is a text file it should use utf-8 encoding; got:"
                f" {encoding!r}"
            ),
        ):
            report.write_html(file)

    with open(tmp_file_path, "r", encoding="latin-1") as file:
        saved_content = file.read()
    assert "</html>" not in saved_content


@skip_polars_installed_without_pyarrow
def test_verbosity_parameter(df_module, capsys):
    df = df_module.make_dataframe(
        dict(
            a=[1, 2, 3, 4],
            b=["one", "two", "three", "four"],
            c=[11.1, 11.2, 11.3, 11.4],
        )
    )

    report = TableReport(df)
    report.html()
    assert capsys.readouterr().err != ""

    report_2 = TableReport(df, verbose=0)
    report_2.html()
    assert capsys.readouterr().err == ""

    report_3 = TableReport(df, verbose=1)
    report_3.html()
    assert capsys.readouterr().err != ""


@skip_polars_installed_without_pyarrow
def test_write_to_stderr(df_module, capsys):
    df = df_module.make_dataframe(
        dict(
            a=[1, 2, 3, 4],
            b=["one", "two", "three", "four"],
            c=[11.1, 11.2, 11.3, 11.4],
        )
    )

    report = TableReport(df)
    report.html()

    captured = capsys.readouterr()

    pattern = re.compile(r"Processing column\s+\d+\s*/\s*\d+")

    assert not re.search(pattern, captured.out)
    assert re.search(pattern, captured.err)


@skip_polars_installed_without_pyarrow
def test_max_plot_columns_parameter(df_module):
    df = df_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(10)}
    )
    summary = TableReport(df)._summary
    assert not summary["plots_skipped"]

    df2 = df_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(30)}
    )
    summary = TableReport(df2)._summary
    assert not summary["plots_skipped"]

    df3 = df_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(31)}
    )
    summary = TableReport(df3)._summary
    assert summary["plots_skipped"]

    df4 = df_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(12)}
    )
    summary = TableReport(df4, max_plot_columns=10)._summary
    assert summary["plots_skipped"]

    df5 = df_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(12)}
    )
    summary = TableReport(df5, max_plot_columns=15)._summary
    assert not summary["plots_skipped"]

    df6 = df_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(5)}
    )
    summary = TableReport(df6, max_plot_columns=None)._summary
    assert not summary["plots_skipped"]

    summary = TableReport(df6, max_plot_columns="all")._summary
    assert not summary["plots_skipped"]


def test_minimal_mode(pd_module):
    # Check that flags are set properly and that the panels are not created
    df = pd_module.example_dataframe
    report = TableReport(df)
    report._set_minimal_mode()
    html = report.html()
    assert "data-test-plots-skipped" in html
    assert "data-test-associations-skipped" in html
    assert 'id="column-summaries-panel"' not in html
    assert 'id="column-associations-panel"' not in html


def test_error_input_type(simple_df, simple_series):
    df = Bunch(X=simple_df, y=simple_series)
    with pytest.raises(TypeError):
        TableReport(df)


@skip_polars_installed_without_pyarrow
def test_single_column_report(df_module):
    # Check that single column report works
    single_col = df_module.example_column
    report = TableReport(single_col)
    col_name = sbd.name(single_col)
    html = report.html()
    assert col_name in html


def test_error_make_table():
    # Make codecov happy
    with pytest.raises(TypeError, match="Expecting a Pandas or Polars DataFrame"):
        make_table(np.array([1]))


@pytest.mark.parametrize("arg", ["max_plot_columns", "max_association_columns"])
def test_bad_cols_parameter(pd_module, arg):
    df = pd_module.example_dataframe
    with pytest.raises(ValueError):
        TableReport(df, **{arg: -1})


def test_array_dim_check():
    array_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert array_3d.ndim == 3
    with pytest.raises(ValueError, match=r"Input (NumPy )?array has 3 dimensions"):
        TableReport(array_3d)

    array_4d = np.array([[[[1]]]])
    assert array_4d.ndim == 4
    with pytest.raises(ValueError, match=r"Input (NumPy )?array has 4 dimensions"):
        TableReport(array_4d)

    array_1d = np.array([1, 2, 3])
    assert array_1d.ndim == 1

    TableReport(array_1d)


numpy_test_cases = [
    (
        np.array(
            [
                [1, 2, 3],
                [
                    4,
                    5,
                    6,
                ],
            ]
        ),
        3,
    ),
    (np.array([[10, 20], [30, 40], [50, 60], [60, 70]]), 2),
]


@pytest.mark.parametrize("input_array, expected_columns", numpy_test_cases)
def test_numpy_array_columns(input_array, expected_columns):
    report = TableReport(input_array, max_association_columns=0)

    assert report._summary["n_columns"] == expected_columns


def _pyarrow_available():
    try:
        import pyarrow  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.xfail(
    condition=_pyarrow_available(),
    reason="Test expects pyarrow to not be installed, but it is installed",
)
def test_polars_df_no_pyarrow():
    # Test that when using a Polars dataframe without pyarrow installed,
    # the appropriate flag is set in the summary and the message appears in the HTML.
    pl = pytest.importorskip("polars")

    df = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": ["a", "b", "c", "d", "e"],
            "C": [10, 20, 30, 40, 50],
        }
    )

    report = TableReport(df, verbose=0)
    summary = report._summary

    assert summary.get("associations_skipped_polars_no_pyarrow", False) is True
    assert summary.get("dataframe_module", "") == "polars"

    html_snippet = report.html_snippet()
    assert (
        "Computing pairwise associations is not available for Polars dataframes "
        "when PyArrow is not installed" in html_snippet
    )
