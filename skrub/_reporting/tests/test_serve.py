import webbrowser
from urllib.request import urlopen

from skrub._reporting._serve import open_in_browser


class UrlOpener:
    def __call__(self, url):
        with urlopen(url) as f:
            self.content = f.read()


def test_open_in_browser_file(monkeypatch):
    opener = UrlOpener()
    monkeypatch.setattr(webbrowser, "open", opener)
    open_in_browser("hello")
    assert opener.content == b"hello"
