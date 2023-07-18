import os
import tempfile
from pathlib import Path

from skrub.datasets._utils import clear_data_home, get_data_dir, get_data_home


def test_get_data_dir():
    """
    Tests function ``get_data_dir()``.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        expected_return_value_default = Path(tmpdirname)
        assert get_data_dir(data_home=tmpdirname) == expected_return_value_default

        expected_return_value_custom = expected_return_value_default / "tests"
        assert (
            get_data_dir(name="tests", data_home=tmpdirname)
            == expected_return_value_custom
        )


def test_get_data_home_str():
    """
    Test function for ``get_data_home()`` when data_home is a string.
    """
    with tempfile.TemporaryDirectory() as tmpdirname:
        # get_data_home will point to a pre-existing folder
        assert os.path.exists(tmpdirname)
        data_home = get_data_home(data_home=tmpdirname)
        assert data_home == tmpdirname
        assert os.path.exists(data_home)

        # clear_data_home will delete both the content and the folder itself
        clear_data_home(data_home=data_home)
        assert not os.path.exists(data_home)

        # if the folder is missing it will be created again
        assert not os.path.exists(tmpdirname)
        data_home = get_data_home(data_home=tmpdirname)
        assert os.path.exists(data_home)


def test_get_data_home_None():
    """
    Test function for ``get_data_home()`` with data_home=None.
    """
    # get path of the folder 'skrub_data' in the user home folder
    user_path = os.path.join("~", "skrub_data")
    user_path = os.path.expanduser(user_path)

    # if the folder does not exist, create it, and clear it
    if not os.path.exists(user_path):
        data_home = get_data_home(data_home=None)
        assert data_home == user_path
        assert os.path.exists(data_home)

        clear_data_home()
        assert not os.path.exists(data_home)
