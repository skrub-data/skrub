import pathlib
import webbrowser
from urllib.request import urlopen

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


@pytest.fixture
def browser_mock_bad_request(monkeypatch):
    opener = BadUrlOpener()
    monkeypatch.setattr(webbrowser, "open", opener)
    return opener
