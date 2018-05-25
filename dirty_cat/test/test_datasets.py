"""
Test the datasets module
"""
# Author: Pierre Glaser
# -*- coding: utf-8 -*-
import os
from tempfile import mkstemp

import dirty_cat.datasets as datasets
import dirty_cat.utils as utils

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')
tmpdir = None
url_request = None
file_mock = None


def setup_mock(true_module=datasets, mock_module=utils):
    global original_url_request
    global mock_url_request
    mock_url_request = mock_module.mock_request_get
    original_url_request = true_module.request_get
    true_module.request_get = mock_url_request


def teardown_mock(true_module=datasets):
    global original_url_request
    true_module.request_get = original_url_request


@utils.with_setup(setup=setup_mock, teardown=teardown_mock)
def test_fetch_file_overwrite():
    # overwrite non-exiting file.
    test_dir = datasets.get_data_dir(name='test')
    fil = datasets._fetch_file(url='http://foo/', data_dir=test_dir,
                               overwrite=True
                               , uncompress=False)

    assert os.path.exists(fil)
    with open(fil, 'r') as fp:
        assert fp.read() == ' '

    # Modify content
    with open(fil, 'w') as fp:
        fp.write('some content')

    # Don't overwrite existing file.
    fil = datasets._fetch_file(url='http://foo/', data_dir=test_dir,
                               overwrite=False, uncompress=False)
    assert os.path.exists(fil)
    with open(fil, 'r') as fp:
        assert fp.read() == 'some content'

    # Overwrite existing file.
    # Overwrite existing file.
    fil = datasets._fetch_file(url='http://foo/', data_dir=test_dir,
                               overwrite=True, uncompress=False)
    assert os.path.exists(fil)
    with open(fil, 'r') as fp:
        assert fp.read() == ' '


def test_md5_sum_file():
    # Create dummy temporary file
    out, f = mkstemp()
    os.write(out, b'abcfeg')
    os.close(out)
    assert datasets._md5_sum_file(f) == '18f32295c556b2a1a3a8e68fe1ad40f7'
    os.remove(f)

# if __name__ == '__main__':
#     test_fetch_file_overwrite()
