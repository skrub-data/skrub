from urllib.request import urlopen
import webbrowser

from skrubview._serve import open_in_browser


class UrlOpener:
    def __call__(self, url):
        with urlopen(url) as f:
            self.content = f.read()


def test_open_in_browser_file(monkeypatch):
    opener = UrlOpener()
    monkeypatch.setattr(webbrowser, "open", opener)
    open_in_browser("hello")
    assert opener.content == b"hello"
