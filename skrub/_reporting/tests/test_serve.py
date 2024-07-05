import pytest

from skrub._reporting._serve import open_in_browser


def test_open_in_browser(browser_mock):
    open_in_browser("hello")
    assert browser_mock.content == b"hello"


def test_open_in_browser_failure(browser_mock_bad):
    with pytest.raises(RuntimeError, match="Failed to open report"):
        open_in_browser("hello")
