import os
import tempfile
import time

import pytest

from skrub._data_ops import _utils


def test_prune_folder_with_standard_name_dirs():
    eight_days_ago = time.time() - 8 * 24 * 3600
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            dirname = os.path.join(tmpdir, f"full_data_op_report_{i}")
            os.mkdir(dirname)
            # setting the access time of the first two files 
            # to 8 days ago, so that they should be pruned
            if i < 2:
                os.utime(dirname, (eight_days_ago, eight_days_ago))

        assert len(os.listdir(tmpdir)) == 3

        _utils.prune_folder(tmpdir)

        remaining_items = os.listdir(tmpdir)
        assert len(remaining_items) == 1


def test_prune_folder_with_nonstandard_name_dirs():
    eight_days_ago = time.time() - 8 * 24 * 3600
    with tempfile.TemporaryDirectory() as tmpdir:
        dirname = os.path.join(tmpdir, "other_report")
        os.mkdir(dirname)
        # setting access time to 8 days ago
        os.utime(dirname, (eight_days_ago, eight_days_ago))

        assert len(os.listdir(tmpdir)) == 1

        _utils.prune_folder(tmpdir)

        remaining_items = os.listdir(tmpdir)
        assert len(remaining_items) == 1
        assert remaining_items[0] == "other_report"


def test_prune_folder_catch_exception(monkeypatch):
    eight_days_ago = time.time() - 8 * 24 * 3600

    with tempfile.TemporaryDirectory() as tmpdir:
        dirname = os.path.join(tmpdir, "full_data_op_report_test")
        os.mkdir(dirname)
        os.utime(dirname, (eight_days_ago, eight_days_ago))

        def mock_rmtree(path, *args, **kwargs):
            raise OSError("Cannot delete folder")

        monkeypatch.setattr("shutil.rmtree", mock_rmtree)

        assert len(os.listdir(tmpdir)) == 1

        with pytest.warns(UserWarning, match="Could not delete"):
            _utils.prune_folder(tmpdir)

        monkeypatch.undo()
