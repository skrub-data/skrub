"""
Test the datasets module
"""
# Author: Pierre Glaser
# -*- coding: utf-8 -*-
import os
from tempfile import mkstemp
import shutil
import zipfile
import contextlib

import dirty_cat.datasets.utils as datasets_utils
import dirty_cat.datasets.tests.utils as utils

currdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(currdir, 'data')
tmpdir = None
url_request = None
file_mock = None


def setup_mock(true_module=datasets_utils, mock_module=utils):
    global original_url_request
    global mock_url_request
    mock_url_request = mock_module.mock_request_get
    original_url_request = true_module.request_get
    true_module.request_get = mock_url_request


def teardown_mock(true_module=datasets_utils):
    global original_url_request
    true_module.request_get = original_url_request


@utils.with_setup(setup=setup_mock, teardown=teardown_mock)
def test_fetch_file_overwrite():
    utils.MockResponse.set_with_content_length(True)
    utils.MockResponse.set_to_zipfile(False)
    test_dir = datasets_utils.get_data_dir(name='test')
    from dirty_cat.datasets import fetching
    try:
        # test that filename is a md5 hash of the url if
        # the url ends with /
        fil = fetching._fetch_file(url='http://foo/', data_dir=test_dir,
                                   overwrite=True, uncompress=False,
                                   show_progress=False)
        assert os.path.basename(fil) == datasets_utils.md5_hash('/')
        os.remove(fil)

        # overwrite non-exiting file.
        fil = fetching._fetch_file(url='http://foo/testdata', data_dir=test_dir,
                                   overwrite=True, uncompress=False,
                                   show_progress=False)

        # check if data_dir is actually used
        assert os.path.dirname(fil) == test_dir
        assert os.path.exists(fil)
        with open(fil, 'r') as fp:
            assert fp.read() == ' '

        # Modify content
        with open(fil, 'w') as fp:
            fp.write('some content')

        # Don't overwrite existing file.
        fil = fetching._fetch_file(url='http://foo/testdata', data_dir=test_dir,
                                   overwrite=False, uncompress=False,
                                   show_progress=False)
        assert os.path.exists(fil)
        with open(fil, 'r') as fp:
            assert fp.read() == 'some content'

        # Overwrite existing file.
        # Overwrite existing file.
        fil = fetching._fetch_file(url='http://foo/testdata', data_dir=test_dir,
                                   overwrite=True, uncompress=False,
                                   show_progress=False)
        assert os.path.exists(fil)
        with open(fil, 'r') as fp:
            assert fp.read() == ' '

        # modify content,
        # change filename,add it in argument, and set overwrite to false
        with open(fil, 'w') as fp:
            fp.write('some content')
        newf = 'moved_file'
        os.rename(fil, os.path.join(test_dir, newf))
        fetching._fetch_file(url='http://foo/testdata',
                             filenames=(newf,),
                             data_dir=test_dir,
                             overwrite=False, uncompress=False,
                             show_progress=False)
        assert (
            not os.path.exists(fil))  # it has been removed and should not have
        with open(os.path.join(test_dir, newf), 'r') as fp:
            assert fp.read() == 'some content'
        # been downloaded again
        os.remove(os.path.join(test_dir, newf))

        # # create a zipfile with a file inside, remove the file, and
        #
        # zipd = os.path.join('testzip.zip')
        # with contextlib.closing(zipfile.ZipFile())
        #     fetching._fetch_file(url='http://foo/', filenames=('test_filename',),
        #                          data_dir=test_dir,
        #                          overwrite=False, uncompress=False,
        #                          show_progress=False)

        # add wrong md5 sum file and catch ValueError
        try:
            fil = fetching._fetch_file(url='http://foo/testdata',
                                       data_dir=test_dir,
                                       overwrite=True, uncompress=False,
                                       show_progress=False, md5sum='1')
            raise ValueError  # if no error raised in the previous line,
            #  it is bad:
            # a wrong md5 sum should raise an error. So forcing the except chunk
            # to happen anyway
        except Exception as e:
            assert isinstance(e, fetching.FileChangedError)
        utils.MockResponse.set_with_content_length(False)
        # write content if no content size
        fil = fetching._fetch_file(url='http://foo/testdata', data_dir=test_dir,
                                   overwrite=True, uncompress=False,
                                   show_progress=False)
        assert os.path.exists(fil)
        os.remove(fil)
    finally:
        shutil.rmtree(test_dir)


@utils.with_setup(setup=setup_mock, teardown=teardown_mock)
def test_convert_file_to_utf8(monkeypatch):
    from dirty_cat.datasets import fetching
    datadir = os.path.join(datasets_utils.get_data_dir(),
                           'testdata')
    try:
        # Specify some content encoded in latin-1, and make sure the final file
        # contains the same content, but in utf-8. Here, '\xe9' in latin-1 is
        # '\xc3\xa9' in utf-8

        with monkeypatch.context() as m:
            m.setattr(utils.MockResponse, "zipresult", False)
            m.setattr(utils.MockResponse, "_file_contents", b'\xe9')

            dataset_info = fetching.DatasetInfo(
                name='testdata',
                urlinfos=(fetching.UrlInfo(
                    url='http://foo/data',
                    filenames=('data',),
                    uncompress=False,
                    encoding='latin-1'),),
                main_file='data',
                source='http://foo/')

            info = fetching.fetch_dataset(dataset_info, show_progress=False)

            with open(info['path'], 'rb') as f:
                content = f.read()

            assert content == b'\xc3\xa9'
            os.unlink(info['path'])

            m.setattr(utils.MockResponse, "zipresult", True)

            dataset_info_with_zipfile = fetching.DatasetInfo(
                name='testdata',
                urlinfos=(fetching.UrlInfo(
                    url='http://foo/data.zip',
                    filenames=('unzipped_data.txt',),
                    uncompress=True,
                    encoding='latin-1'),),
                main_file='unzipped_data.txt',
                source='http://foo/')

            info_unzipped = fetching.fetch_dataset(
                dataset_info_with_zipfile, show_progress=False)

            with open(info_unzipped['path'], 'rb') as f:
                content_unzipped = f.read()

            assert content_unzipped == b'\xc3\xa9'

    finally:
        if os.path.exists(datadir):
            shutil.rmtree(datadir)


def test_md5_sum_file():
    # Create dummy temporary file
    out, f = mkstemp()
    os.write(out, b'abcfeg')
    os.close(out)
    assert datasets_utils._md5_sum_file(f) == '18f32295c556b2a1a3a8e68fe1ad40f7'
    os.remove(f)


@utils.with_setup(setup=setup_mock, teardown=teardown_mock)
def test_fetch_dataset():
    from dirty_cat.datasets import fetching
    utils.MockResponse.set_with_content_length(True)
    datadir = os.path.join(datasets_utils.get_data_dir(),
                           'testdata')
    try:
        urlinfo = fetching.DatasetInfo(name='testdata',
                                       urlinfos=(fetching.UrlInfo(
                                           url='http://foo/data',
                                           filenames=('data',),
                                           uncompress=False,
                                           encoding='utf-8'),),
                                       main_file='data',
                                       source='http://foo/')
        fetching.fetch_dataset(urlinfo, show_progress=False)
        assert os.path.exists(os.path.join(datadir, 'data'))
        shutil.rmtree(os.path.join(datadir))

        # test with zipped data
        utils.MockResponse.set_to_zipfile(True)
        utils.MockResponse.set_with_content_length(False)
        urlinfo = fetching.DatasetInfo(name='testdata',
                                       urlinfos=(fetching.UrlInfo(
                                           url='http://foo/data.zip',
                                           filenames=('unzipped_data.txt',),
                                           uncompress=True,
                                           encoding='utf-8'),),
                                       main_file='unzipped_data.txt',
                                       source='http://foo/')
        fetching.fetch_dataset(urlinfo, show_progress=False)
        assert os.path.exists(os.path.join(datadir, 'unzipped_data.txt'))
    finally:
        if os.path.exists(datadir):
            shutil.rmtree(datadir)


if __name__ == '__main__':
    # pass
    # test_fetch_file_overwrite()
    test_fetch_dataset()
