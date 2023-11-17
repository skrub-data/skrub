import pandas
import pytest

DATAFRAME_MODULES = [pandas]
try:
    import polars

    DATAFRAME_MODULES.append(polars)
except ImportError:
    pass


@pytest.fixture(params=DATAFRAME_MODULES)
def px(request):
    return request.param
