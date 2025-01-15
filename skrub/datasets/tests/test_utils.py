import shutil
import tempfile
from importlib import reload
from pathlib import Path

import pytest

from skrub.datasets import _utils


@pytest.mark.parametrize("data_home_type", ["string", "path"])
def test_get_data_dir(data_home_type):
    with tempfile.TemporaryDirectory() as target_dir:
        tmp_dir = target_dir if data_home_type == "string" else Path(target_dir)

        # with a pre-existing folder
        assert Path(target_dir).exists()
        data_home = _utils.get_data_home(data_home=tmp_dir)
        assert data_home == Path(target_dir).resolve()

        assert data_home.exists()

        assert (
            _utils.get_data_dir(name="tests", data_home=tmp_dir)
            == data_home.resolve() / "tests"
        )

        # if the folder is missing it will be created
        shutil.rmtree(tmp_dir)
        assert not Path(target_dir).exists()
        data_home = _utils.get_data_home(data_home=tmp_dir)
        assert data_home.exists()


def test_get_data_home_default():
    """Test function for ``get_data_home()`` with default `data_home`."""
    # We should take care of not deleting the folder if our user
    # already cached some data
    user_path = Path("~").expanduser() / "skrub_data"
    is_already_existing = user_path.exists()

    data_home = _utils.get_data_home(data_home=None)
    assert data_home == user_path
    assert data_home.exists()

    if not is_already_existing:
        # Clear the folder if it was not already existing.
        shutil.rmtree(user_path)
        assert not user_path.exists()


def test_get_data_home_with_parameter(tmp_path):
    dirpath = tmp_path / "my-home"

    assert _utils.get_data_home(dirpath) == dirpath
    assert dirpath.exists()


def test_get_data_home_with_envar(monkeypatch, tmp_path):
    dirpath = tmp_path / "my-home"

    try:
        with monkeypatch.context() as mp:
            mp.setenv(_utils.DATA_HOME_ENVAR_NAME, str(dirpath))
            reload(_utils)

            assert _utils.get_data_home() == dirpath
            assert dirpath.exists()
    finally:
        reload(_utils)
