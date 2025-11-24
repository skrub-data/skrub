import datetime
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from skrub._data_ops import _utils
from skrub._utils import random_string


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# Fixture to create directories with optional old timestamp
@pytest.fixture
def create_dir(tmp_dir):
    def _create(name=None, days_old=None):
        if name is None:
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H%M%S")
            name = f"full_data_op_report_{now}_{random_string()}"
        path = tmp_dir / name
        path.mkdir()
        if days_old is not None:
            old_time = time.time() - days_old * 24 * 3600
            # setting access and modified times to old_time
            os.utime(path, (old_time, old_time))
        return path

    return _create


def test_prune_directory_with_standard_name_dirs(tmp_dir, create_dir):
    # Making a directory older than 7 days that should be pruned
    # and one recent directory that should not be pruned
    create_dir(name=None, days_old=None)
    create_dir(name=None, days_old=8)

    assert len(list(tmp_dir.iterdir())) == 2

    _utils.prune_directory(tmp_dir)

    remaining_items = list(tmp_dir.iterdir())
    assert len(remaining_items) == 1


def test_prune_directory_with_nonstandard_name_dirs(tmp_dir, create_dir):
    # Making a directory older than 7 days with a non-matching name
    # so it should not be pruned
    create_dir("other_report", days_old=8)

    assert len(list(tmp_dir.iterdir())) == 1

    _utils.prune_directory(tmp_dir)

    remaining_items = list(tmp_dir.iterdir())
    assert len(remaining_items) == 1
    assert remaining_items[0].name == "other_report"


def test_prune_directory_catch_exception(tmp_dir, create_dir, monkeypatch):
    create_dir(name=None, days_old=8)

    def mock_rmtree(path, *args, **kwargs):
        raise OSError("Cannot delete folder")

    monkeypatch.setattr(shutil, "rmtree", mock_rmtree)

    assert len(list(tmp_dir.iterdir())) == 1

    with pytest.warns(UserWarning, match="Could not delete"):
        _utils.prune_directory(tmp_dir)

    assert len(list(tmp_dir.iterdir())) == 1

    monkeypatch.undo()
