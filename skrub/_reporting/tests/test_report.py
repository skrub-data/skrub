import pathlib

import polars as pl

from skrubview import Report

def test_report():
    data_dir = pathlib.Path(__file__).parent / 'data'
    fname = 'air_quality_no2_long.parquet'
    data_file =  data_dir / fname
    df = pl.read_parquet(data_file)
    report = Report(df)
    report.html
    report.html_snippet
    report.json
    report.text
    report._any_summary
    report._repr_mimebundle_()
