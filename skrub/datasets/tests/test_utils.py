import shutil
import sys
import tempfile
from pathlib import Path

import pytest

from skrub.datasets._utils import DATA_HOME_ENVAR_NAME, get_data_dir, get_data_home


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
