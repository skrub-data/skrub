import pytest

from skrub._dataframe._polars import rename_columns
from skrub.conftest import _POLARS_INSTALLED

if _POLARS_INSTALLED:
    import polars as pl


@pytest.mark.skipif(not _POLARS_INSTALLED, reason="Polars is not available")
def test_rename_columns():
    df = pl.DataFrame({"a column": [1], "another": [1]})
    df = rename_columns(df, str.swapcase)
    assert list(df.columns) == ["A COLUMN", "ANOTHER"]
