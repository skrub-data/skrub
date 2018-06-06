import sys
import os
import requests
from functools import wraps
import hashlib
import zipfile
import shutil
import contextlib
import tarfile


def get_data_dir(name=None):
    """ Returns the directories in which dirty_cat looks for data.

    This is typically useful for the end-user to check where the data is
    downloaded and stored.

    """
    # assuming we are in datasets.utils, this calls the module
    module_path = os.path.dirname(os.path.dirname(__file__))
    data_dir = os.path.join(module_path, 'data')
    if name is not None:
        data_dir = os.path.join(data_dir, name)
    return data_dir


@wraps(requests.get)
def request_get(*args, **kwargs):
    return requests.get(*args, **kwargs)


def md5_hash(string):
    m = hashlib.md5()
    m.update(string.encode('utf-8'))
    return m.hexdigest()


def _md5_sum_file(path):
    """ Calculates the MD5 sum of a file.
    """
    with open(path, 'rb') as f:
        m = hashlib.md5()
        while True:
            data = f.read(8192)
            if not data:
                break
            m.update(data)
    return m.hexdigest()


def _check_if_exists(path, remove=False):
    if remove:
        try:
            os.remove(path)
        except OSError:
            pass
        return False
    else:
        return os.path.exists(path)


def _uncompress_file(file_, delete_archive=True):
    """Uncompress files contained in a data_set.


    Parameters
    ----------
    file_: path to file
    delete_archive: whether to delete the compressed file afterwards


    Returns
    -------
    None if everything worked out fine
    ValueError otherwise


    Notes
    -----
    only supports zip and gzip
    """
    sys.stderr.write('Extracting data from %s...' % file_)
    data_dir = os.path.dirname(file_)
    # We first try to see if it is a zip file
    try:
        filename, ext = os.path.splitext(file_)
        with open(file_, "rb") as fd:
            header = fd.read(4)
        processed = False
        if zipfile.is_zipfile(file_):
            z = zipfile.ZipFile(file_)
            z.extractall(path=data_dir)
            z.close()
            if delete_archive:
                os.remove(file_)
            file_ = filename
            processed = True
        elif ext == '.gz' or header.startswith(b'\x1f\x8b'):
            import gzip
            gz = gzip.open(file_)
            if ext == '.tgz':
                filename = filename + '.tar'
            out = open(filename, 'wb')
            shutil.copyfileobj(gz, out, 8192)
            gz.close()
            out.close()
            # If file is .tar.gz, this will be handle in the next case
            if delete_archive:
                os.remove(file_)
            file_ = filename
            processed = True
        if os.path.isfile(file_) and tarfile.is_tarfile(file_):
            with contextlib.closing(tarfile.open(file_, "r")) as tar:
                tar.extractall(path=data_dir)
            if delete_archive:
                os.remove(file_)
            processed = True
        if not processed:
            raise IOError(
                "[Uncompress] unknown archive file format: %s" % file_)

        sys.stderr.write('.. done.\n')
    except Exception as e:
        print('Error uncompressing file: %s' % e)
        raise
