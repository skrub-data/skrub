import os
import requests
from io import BytesIO
import zipfile
import contextlib
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
    with_content_length = True
    zipresult = False
    _file_contents = ' '.encode('utf-8')

    @classmethod
    def set_with_content_length(cls, b):
        """
        set the with_total_length attribute
        Parameters
        ----------
        b: if True, include total_length in the response

        Returns
        -------

        """

        # I added a classmethod for this instead of
        # changing the syntax of request_get by adding
        # a with_total_length in argument that would break the compatibility
        # with request.get
        cls.with_content_length = b

    @classmethod
    def set_to_zipfile(cls, b):
        cls.zipresult = b

    def __init__(self, url, params, stream=True):
        super(MockResponse, self).__init__()
        self.url = url
        self.params = params
        if self.zipresult:
            # StringIO vs BytesIO: using BytesIO in accorance with the
            # open(file,'wb') from the _fetch_file function
            self.raw = BytesIO()
            with zipfile.ZipFile(self.raw, mode='w',
                                 compression=zipfile.ZIP_DEFLATED) as mf:
                mf.writestr('unzipped_data.txt', self._file_contents)

            # go to the start of the stream
            self.raw.seek(0)
            # self.raw = self.raw.wr(file=self.raw)
            if stream:
                self._content = self.raw.getvalue()

        else:
            if stream:
                self.raw = BytesIO(self._file_contents)
            else:
                self._content = self._file_contents

        if self.with_content_length:
            if self.zipresult:
                self.headers['Content-Length'] = len(self.raw.getvalue())
                self.total_length = len(self.raw.getvalue())
            else:
                self.headers['Content-Length'] = len(self._file_contents)
                self.total_length = len(self._file_contents)

    def iter_content(self, chunk_size=1, decode_unicode=False):
        """

        it is actually pretty hard to read from a streamed
        zipped file-like object.2 possibilites:
            - force non-stream reading when using zipfiles
            - create a custom iter_content method to handle this case
        Parameters
        ----------
        chunk_size
        decode_unicode

        Returns
        -------

        """

        print('iterating from mocked response')
        return super(MockResponse, self).iter_content(chunk_size=chunk_size,
                                                      decode_unicode=decode_unicode)


def mock_request_get(url, params=None, stream=True):
    # should we be able to input a specialized content for the mockrequest?
    return MockResponse(url, params=None)
