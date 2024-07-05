import pytest

from skrub._reporting._serve import open_in_browser


def test_open_in_browser(browser_mock):
    open_in_browser("hello")
    assert browser_mock.content == b"hello"


def test_open_in_browser_bad_request(browser_mock_bad_request):
    with pytest.raises(Exception, match="File not found"):
        open_in_browser("hello")


def test_open_in_browser_failure(browser_mock_no_request):
    with pytest.raises(RuntimeError, match="Failed to open report"):
        open_in_browser("hello")
