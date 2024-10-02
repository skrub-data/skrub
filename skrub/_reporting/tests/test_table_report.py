import json
import re

from skrub import TableReport
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
    assert "Columns with null values" in html
    assert "First 10 columns" in html
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
    assert report._any_summary["title"] == "the title"
    del report._summary_with_plots
    assert report._any_summary["title"] == "the title"
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
