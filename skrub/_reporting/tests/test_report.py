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
    new_report_id = get_report_id(report.html())
    assert len(new_report_id) == 8
    assert report_id != new_report_id
    report.html_snippet()
    report.json()
    report._any_summary
    report._repr_mimebundle_()
    report._repr_html_()


def test_report_few_columns(df_module):
    report = TableReport(df_module.example_dataframe)
    assert "First 10 columns" not in report.html()


def test_report_few_rows(df_module):
    df = sbd.slice(df_module.example_dataframe, 2)
    TableReport(df).html()


def test_report_empty_dataframe(df_module):
    html = TableReport(df_module.empty_dataframe).html()
    assert "The dataframe is empty." in html
