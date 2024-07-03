import json
import pathlib

import pandas as pd

from skrub._reporting import TableReport
from skrub._reporting._summarize import summarize_dataframe


def test_summarize():
    data_dir = pathlib.Path(__file__).parent / "data"
    fname = "air_quality_no2_long.parquet"
    data_file = data_dir / fname
    df = pd.read_parquet(data_file)
    expected = json.loads((data_dir / f"{fname}.expected.json").read_text("utf-8"))
    assert json.loads(TableReport(df, title=fname).json) == expected
    summary = summarize_dataframe(df, with_plots=True)
    for c in summary["columns"]:
        assert c["value_is_constant"] or len(c["plot_names"]) == 1
    summary = summarize_dataframe(df, with_plots=True, order_by="date.utc")
    for c in summary["columns"]:
        assert c["value_is_constant"] or len(c["plot_names"]) == 1
