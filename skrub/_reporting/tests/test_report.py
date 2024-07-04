import pathlib

import pandas as pd
import pytest

from skrub import TableReport


def test_report():
    data_dir = pathlib.Path(__file__).parent / "data"
    fname = "air_quality_no2_long.parquet"
    data_file = data_dir / fname
    try:
        df = pd.read_parquet(data_file)
    except ImportError:
        pytest.skip("missing pyarrow, cannot read parquet")
    report = TableReport(df)
    report.html
    report.html_snippet
    report.json
    report._any_summary
    report._repr_mimebundle_()
    report._repr_html_()
