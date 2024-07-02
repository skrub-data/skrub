import pathlib

import pandas as pd

from skrub import Report


def test_report():
    data_dir = pathlib.Path(__file__).parent / "data"
    fname = "air_quality_no2_long.parquet"
    data_file = data_dir / fname
    df = pd.read_parquet(data_file)
    report = Report(df)
    report.html
    report.html_snippet
    report.json
    report._any_summary
    report._repr_mimebundle_()
