import contextlib
import datetime
import json
import re
import warnings
from pathlib import Path

import pytest

from skrub import TableReport, ToDatetime
from skrub import _dataframe as sbd


def get_report_id(html):
    return re.search(r'<skrub-table-report.*?id="report_([a-z0-9]+)"', html).group(1)


def test_report(air_quality):
    col_filt = {
        "first_2": {
            "display_name": "First 2",
            "columns": sbd.column_names(air_quality)[:2],
        }
    }
    report = TableReport(air_quality, title="the title", column_filters=col_filt)
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
    assert report._get_summary()["title"] == "the title"
    del report._summary_with_plots
    assert report._get_summary()["title"] == "the title"
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


def test_few_columns(df_module, check_polars_numpy2):
    report = TableReport(df_module.example_dataframe)
    assert "First 10 columns" not in report.html()


def test_few_rows(df_module, check_polars_numpy2):
    df = sbd.slice(df_module.example_dataframe, 2)
    TableReport(df).html()


def test_empty_dataframe(df_module):
    html = TableReport(df_module.empty_dataframe).html()
    assert "The dataframe is empty." in html


def test_open(pd_module, browser_mock):
    TableReport(pd_module.example_dataframe, title="the title").open()
    assert b"the title" in browser_mock.content


def test_non_hashable_values(df_module):
    # non-regression test for #1066
    df = df_module.make_dataframe(dict(a=[[1, 2, 3], None, [4]]))
    html = TableReport(df).html()
    assert "[1, 2, 3]" in html


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


def test_duplicate_columns(pd_module):
    df = pd_module.make_dataframe({"a": [1, 2], "b": [3, 4]})
    df.columns = ["a", "a"]
    TableReport(df).html()


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


def test_max_plot_columns_parameter(df_module):
    df = df_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(10)}
    )
    summary = TableReport(df)._get_summary()
    assert not summary["plots_skipped"]

    df2 = df_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(30)}
    )
    summary = TableReport(df2)._get_summary()
    assert not summary["plots_skipped"]

    df3 = df_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(31)}
    )
    summary = TableReport(df3)._get_summary()
    assert summary["plots_skipped"]

    df4 = df_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(12)}
    )
    summary = TableReport(df4, max_plot_columns=10)._get_summary()
    assert summary["plots_skipped"]

    df5 = df_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(12)}
    )
    summary = TableReport(df5, max_plot_columns=15)._get_summary()
    assert not summary["plots_skipped"]

    df6 = df_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(5)}
    )
    summary = TableReport(df6, max_plot_columns=None)._get_summary()
    assert not summary["plots_skipped"]
