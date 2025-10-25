import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from skrub.datasets._utils import (
    DATA_HOME_ENVAR_NAME,
    _extract_archive,
    get_data_dir,
    get_data_home,
)


@pytest.mark.parametrize("data_home_type", ["string", "path"])
def test_get_data_dir(data_home_type):
    with tempfile.TemporaryDirectory() as target_dir:
        tmp_dir = target_dir if data_home_type == "string" else Path(target_dir)

        # with a pre-existing folder
        assert Path(target_dir).exists()
        data_home = get_data_home(data_home=tmp_dir)
        assert data_home == Path(target_dir).resolve()

        assert data_home.exists()

        assert (
            get_data_dir(name="tests", data_home=tmp_dir)
            == data_home.resolve() / "tests"
        )

        # if the folder is missing it will be created
        shutil.rmtree(tmp_dir)
        assert not Path(target_dir).exists()
        data_home = get_data_home(data_home=tmp_dir)
        assert data_home.exists()


def test_get_data_home_without_parameter(monkeypatch, tmp_path):
    home = tmp_path / "my-home"
    dirpath = home / "skrub_data"

    with monkeypatch.context() as mp:
        mp.setenv("USERPROFILE" if sys.platform == "win32" else "HOME", str(home))

        assert get_data_home() == dirpath
        assert dirpath.exists()


def test_get_data_home_with_parameter(tmp_path):
    dirpath = tmp_path / "my-home"

    assert get_data_home(dirpath) == dirpath
    assert dirpath.exists()


def test_get_data_home_with_envar(monkeypatch, tmp_path):
    dirpath = tmp_path / "my-home"

    with monkeypatch.context() as mp:
        mp.setenv(DATA_HOME_ENVAR_NAME, str(dirpath))

        assert get_data_home() == dirpath
        assert dirpath.exists()


def test_extract_archive_exception_unlink_called():
    dataset_dir = MagicMock()
    dataset_dir.name = "test_dataset"
    archive_path = MagicMock(spec=Path)
    archive_path.exists.return_value = True

    mock_temp_dir = str(Path("mock_temp_dir"))
    # Patch both tempfile.mkdtemp and shutil.unpack_archive
    # tempfile.mkdtemp and shutil.unpack_archive are used in _extract_archive
    # We simulate unpack_archive raising an exception to test that unlink is called
    with (
        patch("tempfile.mkdtemp", return_value=mock_temp_dir),
        patch("shutil.unpack_archive") as mock_unpack,
    ):
        # Simulate unpack failing
        mock_unpack.side_effect = ValueError("Test error")
        with pytest.raises(ValueError):
            _extract_archive(dataset_dir, archive_path)
        archive_path.unlink.assert_called_once()


def test_extract_archive_unlink_raises():
    dataset_dir = MagicMock()
    dataset_dir.name = "test_dataset"
    archive_path = MagicMock(spec=Path)
    archive_path.exists.return_value = True
    # Setup unlink to fail
    archive_path.unlink.side_effect = OSError("unlink failed")

    mock_temp_dir = str(Path("mock_temp_dir"))
    # Patch both tempfile.mkdtemp and shutil.unpack_archive
    # tempfile.mkdtemp and shutil.unpack_archive are used in _extract_archive
    with (
        patch("tempfile.mkdtemp", return_value=mock_temp_dir),
        patch("shutil.unpack_archive") as mock_unpack,
    ):
        mock_unpack.side_effect = ValueError("Test error")
        # The original error should still be raised
        with pytest.raises(ValueError):
            _extract_archive(dataset_dir, archive_path)
        archive_path.unlink.assert_called_once()
