import pathlib
import webbrowser
from urllib.request import urlopen

import numpy as np
import pytest
from sklearn.utils.fixes import parse_version


@pytest.fixture
def check_polars_numpy2(df_module):
    if df_module.name != "polars":
        return
    pl = df_module.module
    if parse_version(pl.__version__) <= parse_version("1.0.0") and parse_version(
        "2.0.0"
    ) <= parse_version(np.__version__):
        pytest.xfail("polars 1.0.0 does not support numpy 2 causing segfaults")


@pytest.fixture
def data_directory():
    return pathlib.Path(__file__).parent / "data"


@pytest.fixture
def air_quality(df_module, check_polars_numpy2, data_directory):
    data_file = data_directory / "air_quality_tiny.parquet"
    reader = df_module.module.read_parquet
    try:
        df = reader(data_file)
    except ImportError:
        assert df_module.name == "pandas"
        pytest.skip("missing pyarrow, cannot read parquet")
    return df


class UrlOpener:
    def __call__(self, url):
        with urlopen(url) as f:
            self.content = f.read()


@pytest.fixture
def browser_mock(monkeypatch):
    opener = UrlOpener()
    monkeypatch.setattr(webbrowser, "open", opener)
    return opener


@pytest.fixture
def browser_mock_no_request(monkeypatch):
    def opener(url):
        pass

    monkeypatch.setattr(webbrowser, "open", opener)
    return opener


class BadUrlOpener:
    def __call__(self, url):
        with urlopen(url.replace("index.html", "somethingelse")) as f:
            self.content = f.read()


class BadThenGoodUrlOpener:
    def __call__(self, url):
        try:
            with urlopen(url.replace("index.html", "favicon.ico")) as f:
                self.content = f.read()
        except Exception:
            pass
        with urlopen(url) as f:
            self.content = f.read()


@pytest.fixture
def browser_mock_bad_request(monkeypatch):
    opener = BadUrlOpener()
    monkeypatch.setattr(webbrowser, "open", opener)
    return opener


@pytest.fixture
def browser_mock_bad_then_good_request(monkeypatch):
    opener = BadThenGoodUrlOpener()
    monkeypatch.setattr(webbrowser, "open", opener)
    return opener
