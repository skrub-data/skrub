import os
import shutil
from tempfile import mkstemp, mkdtemp
import zipfile
import contextlib
import gzip
import tarfile
from ..utils import _uncompress_file


def test_uncompress():
    # Create a zipfile
    dtemp = mkdtemp()
    ztemp = os.path.join(dtemp, 'test.zip')
    ftemp = 'test'
    try:
        with contextlib.closing(zipfile.ZipFile(ztemp, 'w')) as testzip:
            testzip.writestr(ftemp, 'test')
        _uncompress_file(ztemp)
        assert (os.path.exists(os.path.join(dtemp, ftemp)))
        shutil.rmtree(dtemp)

        dtemp = mkdtemp()
        ztemp = os.path.join(dtemp, 'test.tar')
        ftemp = 'test'

        # Create dummy file in the dtemp folder, so that the finally statement
        # can easily remove it
        fd, temp = mkstemp(dir=dtemp)
        os.close(fd)
        with contextlib.closing(tarfile.open(ztemp, 'w')) as tar:
            tar.add(temp, arcname=ftemp)
        _uncompress_file(ztemp)
        assert (os.path.exists(os.path.join(dtemp, ftemp)))
        shutil.rmtree(dtemp)

        dtemp = mkdtemp()
        ttemp = os.path.join(dtemp, 'test')
        with open(ttemp, 'wb') as f:
            f.write('test'.encode('utf-8'))
        ztemp = os.path.join(dtemp, 'test.gz')
        with open(ttemp, 'rb') as f_in, gzip.open(ztemp, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        _uncompress_file(ztemp)
        # test.gz gets uncompressed into test
        assert (os.path.exists(ttemp))
        shutil.rmtree(dtemp)

        # try to uncompress a corrupted file
        dtemp = mkdtemp()
        ttemp = os.path.join(dtemp, 'test')
        with open(ttemp, 'wb') as fid:
            fid.write('test'.encode('utf-8'))
        try:
            _uncompress_file(ttemp)
        except Exception as e:
            assert isinstance(e, IOError)


    finally:
        # all temp files are created into dtemp except temp
        if os.path.exists(dtemp):
            shutil.rmtree(dtemp)
