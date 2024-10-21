import pandas as pd

from skrub._dataframe._pandas import rename_columns


def test_rename_columns():
    df = pd.DataFrame({"a column": [1], "another": [1]})
    df = rename_columns(df, str.swapcase)
    assert list(df.columns) == ["A COLUMN", "ANOTHER"]
