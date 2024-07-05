import pathlib

import pytest


@pytest.fixture
def air_quality(df_module):
    data_file = pathlib.Path(__file__).parent / "data" / "air_quality_small.parquet"
    reader = df_module.module.read_parquet
    try:
        df = reader(data_file)
    except ImportError:
        assert df_module.name == "pandas"
        pytest.skip("missing pyarrow, cannot read parquet")
    return df
