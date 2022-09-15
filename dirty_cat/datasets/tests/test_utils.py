from pathlib import Path
from unittest import mock


@mock.patch("os.path.dirname")
def test_get_data_dir(mock_os_path_dirname):
    """
    Tests function ``get_data_dir()``.
    """
    from dirty_cat.datasets._utils import get_data_dir

    expected_return_value_default = Path("/user/directory/data")

    mock_os_path_dirname.return_value = "/user/directory/"
    assert get_data_dir() == expected_return_value_default

    expected_return_value_custom = expected_return_value_default / "tests"
    assert get_data_dir("tests") == expected_return_value_custom
