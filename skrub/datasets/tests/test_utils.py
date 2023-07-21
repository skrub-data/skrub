import shutil
import tempfile
from pathlib import Path

from skrub.datasets._utils import get_data_dir, get_data_home


def test_get_data_dir():
    """
    Check the behaviour of `get_data_dir` when passing the path to
    an already existing folder.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirpath = Path(tmpdirname)
        assert get_data_dir(data_home=tmpdirpath) == tmpdirpath

        assert get_data_dir(name="tests", data_home=tmpdirpath) == tmpdirpath / "tests"


def test_get_data_home_str():
    """
    Test function for ``get_data_home()`` when `data_home` is a string.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirpath = Path(tmpdirname)

        # get_data_home will point to a pre-existing folder
        assert tmpdirpath.exists()
        data_home = get_data_home(data_home=tmpdirpath)
        assert data_home == tmpdirpath
        assert data_home.exists()

        # if the folder is missing it will be created
        shutil.rmtree(data_home)
        assert not data_home.exists()
        assert not tmpdirpath.exists()
        data_home = get_data_home(data_home=tmpdirpath)
        assert data_home.exists()


def test_get_data_home_path():
    """
    Test function for ``get_data_home()`` when `data_home` is a pathlib.Path.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirpath = Path(tmpdirname)
        # get_data_home will point to a pre-existing folder
        data_home = get_data_home(data_home=tmpdirname)
        assert data_home == tmpdirpath
        assert data_home.exists()

        # if the folder is missing it will be created
        shutil.rmtree(data_home)
        assert not data_home.exists()
        assert not tmpdirpath.exists()
        data_home = get_data_home(data_home=tmpdirname)
        assert data_home.exists()


def test_get_data_home_default():
    """Test function for ``get_data_home()`` with default `data_home`."""
    # We should take care of not deleting the folder if our user
    # already cached some data
    user_path = Path("~").expanduser() / "skrub_data"
    is_already_existing = user_path.exists()

    data_home = get_data_home(data_home=None)
    assert data_home == user_path
    assert data_home.exists()

    if not is_already_existing:
        # Clear the folder if it was not already existing.
        shutil.rmtree(user_path)
        assert not user_path.exists()
