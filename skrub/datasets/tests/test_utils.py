import shutil
import tempfile
from pathlib import Path

from skrub.datasets._utils import get_data_dir, get_data_home


def test_get_data_dir():
    """
    Tests function ``get_data_dir()``.
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


def test_get_data_home_None():
    """
    Test function for ``get_data_home()`` with `data_home` set to `None`.
    """
    # get path of the folder 'skrub_data' in the user home folder
    user_path = Path("~") / "skrub_data"

    # if the folder does not exist, create it, and clear it
    if not user_path.exists():
        data_home = get_data_home(data_home=None)
        assert data_home == user_path
        assert data_home.exists()

        shutil.rmtree(data_home)
        assert not data_home.exists()
