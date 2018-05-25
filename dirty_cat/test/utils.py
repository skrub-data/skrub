import requests
from io import BytesIO
from functools import wraps


class with_setup:
    def __init__(self, setup, teardown):
        self.setup = setup
        self.teardown = teardown

    def __call__(self, f):
        @wraps(f)
        def func(*args, **kwargs):
            self.setup()
            try:
                f(*args, **kwargs)
            finally:
                self.teardown()

        return func


class MockResponse(requests.Response):
    def __init__(self, url, params, content: str, stream=True):
        super(MockResponse, self).__init__()
        self.url = url
        self.params = params
        if stream:
            # StringIO vs BytesIO: using BytesIO in accorance with the
            # open(file,'wb') from the _fetch_file function
            self.raw = BytesIO(str.encode(content))
        else:
            self._content = content
        self.total_length = len(content)
        self.headers['Content-Length'] = len(content)

    def iter_content(self, chunk_size=1, decode_unicode=False):
        # already exists, just for code readability
        print('iterating from mocked response')
        return super(MockResponse, self).iter_content(chunk_size=chunk_size,
                                                      decode_unicode=decode_unicode)


def mock_request_get(url, params=None, stream=True):
    # should we be able to input a specialized content for the mockrequest?
    return MockResponse(url, params=None, content=' ')
