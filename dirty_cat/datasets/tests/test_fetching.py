"""

Tests fetching.py (datasets fetching off OpenML.org).

"""

# Author: Lilian Boulard <lilian.boulard@inria.fr> || https://github.com/Phaide

import os
import sys
import pytest
import shutil

from unittest import mock
from unittest.mock import mock_open


def get_test_data_dir() -> str:
    from dirty_cat.datasets.utils import get_data_dir
    return get_data_dir("tests")


def test_fetch_openml_dataset():
    """
    Tests the ``fetch_openml_dataset()``
    function in a real environment.
    Though, to avoid the test being too long,
    we will download a small dataset (<1000 entries).

    Reference: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/tests/test_openml.py
    """

    from dirty_cat.datasets.fetching import fetch_openml_dataset
    from urllib.error import URLError

    # Information of the test dataset.
    # Centralizes its information here.
    test_dataset = {
        "id": 50,
        "desc_start": "**Author**",
        "url": "https://www.openml.org/d/50",
        "csv_name": "tic-tac-toe.csv",
        "dataset_rows_count": 958,
        "dataset_columns_count": 10,
    }

    test_data_dir = get_test_data_dir()

    try:  # Try-finally block used to remove the test data directory at the end.
        try:

            # First, we want to purposefully test ValueError exceptions.
            with pytest.raises(ValueError):
                assert fetch_openml_dataset(dataset_id=0, data_directory=test_data_dir)
                assert fetch_openml_dataset(dataset_id=2**32, data_directory=test_data_dir)

            # Valid call
            returned_info = fetch_openml_dataset(dataset_id=test_dataset["id"], data_directory=test_data_dir)

        except URLError:
            # No internet connection, or the website is down.
            # We abort the test.
            #
            # One could try to manually recreate the tree structure
            # created by ``fetch_openml()``,  and the
            # ``.gz`` files within in order to finish the test.
            return

        assert returned_info["description"].startswith(test_dataset["desc_start"])
        assert returned_info["source"] == test_dataset["url"]
        assert returned_info["path"] == os.path.join(test_data_dir, test_dataset["csv_name"])

        assert os.path.isfile(returned_info["path"])

        try:
            from pandas import read_csv, DataFrame
        except ImportError:
            pass
        else:
            # This block will be executed after the end
            # of ``try`` if no error was raised
            # (if pandas.read_csv could be imported).

            dataset: DataFrame = read_csv(filepath_or_buffer=returned_info["path"],
                                          sep=",", quotechar="'", escapechar="\\")

            assert dataset.shape == (test_dataset["dataset_rows_count"],
                                     test_dataset["dataset_columns_count"])

    finally:
        shutil.rmtree(path=test_data_dir, ignore_errors=True)


@mock.patch("os.path.isfile")
@mock.patch("dirty_cat.datasets.fetching._get_features")
@mock.patch("dirty_cat.datasets.fetching._get_details")
@mock.patch("dirty_cat.datasets.fetching._export_gz_data_to_csv")
@mock.patch("dirty_cat.datasets.fetching._download_and_write_openml_dataset")
def test_fetch_openml_dataset_mocked(mock_download, mock_export, mock_get_details, mock_get_features, mock_isfile):
    """
    Test function ``fetch_openml_dataset()``,
    but this time, we mock the functions to
    test its inner mechanisms.
    """

    from dirty_cat.datasets.fetching import fetch_openml_dataset, Details, Features

    mock_get_details.return_value = Details("Dataset_name", "123456", "This is a dummy dataset description.")
    mock_get_features.return_value = Features(["id", "name", "transaction_id", "owner", "recipient"])

    test_data_dir = get_test_data_dir()

    # First, we test the function to see
    # if it behaves correctly when files exists
    mock_isfile.return_value = True

    fetch_openml_dataset(50, test_data_dir)

    mock_download       .assert_not_called()
    mock_export         .assert_not_called()
    mock_get_features   .assert_not_called()
    mock_get_details    .assert_called_once()

    # Reset mocks
    mock_get_details    .reset_mock()

    # This time, the files does not exists
    mock_isfile.return_value = False

    fetch_openml_dataset(50, test_data_dir)

    # Download should be called twice
    mock_download       .assert_called_with(dataset_id=50, data_directory=get_test_data_dir())
    mock_export         .assert_called_once()
    mock_get_features   .assert_called_once()
    mock_get_details    .assert_called_once()


@mock.patch('sklearn.datasets.fetch_openml')
def test__download_and_write_openml_dataset(mock_fetch_openml):
    """Tests function ``_download_and_write_openml_dataset()``."""

    from dirty_cat.datasets.fetching import _download_and_write_openml_dataset

    test_data_dir = get_test_data_dir()
    _download_and_write_openml_dataset(1, test_data_dir)

    mock_fetch_openml.assert_called_once_with(data_id=1, data_home=test_data_dir)


@mock.patch("os.path.isfile")
def test__read_json_from_gz(mock_os_path_isfile):
    """Tests function ``_read_json_from_gz()``."""

    from dirty_cat.datasets.fetching import _read_json_from_gz
    from json import JSONDecodeError

    dummy_file_path = "file/path.gz"

    # Passing an invalid file path (does not exist).
    mock_os_path_isfile.return_value = False
    with pytest.raises(FileNotFoundError):
        assert _read_json_from_gz(dummy_file_path)

    # Passing a valid file path, but reading it
    # does not return JSON-encoded data.
    mock_os_path_isfile.return_value = True
    with mock.patch("gzip.open", mock_open(read_data='This is not JSON-encoded data!')) as _:
        with pytest.raises(JSONDecodeError):
            assert _read_json_from_gz(dummy_file_path)

    # Passing a valid file path, and reading it
    # returns valid JSON-encoded data.
    expected_return_value = {"data": "This is JSON-encoded data!"}
    with mock.patch("gzip.open", mock_open(read_data='{"data": "This is JSON-encoded data!"}')) as _:
        assert _read_json_from_gz(dummy_file_path) == expected_return_value


@mock.patch("dirty_cat.datasets.fetching._read_json_from_gz")
def test__get_details(mock_read_json_from_gz):
    """Tests function ``_get_details()``."""

    from dirty_cat.datasets.fetching import Details, _get_details

    expected_return_value = Details("Dataset_name", "123456", "This is a dummy dataset description.")

    mock_read_json_from_gz.return_value = {
        "data_set_description": {
            "data": "This is JSON data!",
            "name": "Dataset_name",
            "file_id": "123456",
            "description": "This is a dummy dataset description.",
            "extra": "extra_field",
        }
    }

    returned_value = _get_details("/file/name.gz")

    assert returned_value == expected_return_value


@mock.patch("dirty_cat.datasets.fetching._read_json_from_gz")
def test__get_features(mock_read_json_from_gz):
    """Tests function ``_get_features()``."""

    from dirty_cat.datasets.fetching import Features, _get_features

    expected_return_value = Features(["id", "name", "transaction_id", "owner", "recipient"])

    mock_read_json_from_gz.return_value = {
        "data_features": {
            "feature": [
                {"index": "0", "name": "id", "range": "123"},
                {"index": "1", "name": "name", "dub": "dub"},
                {"index": "2", "name": "transaction_id", "type": "5"},
                {"index": "3", "name": "owner", "rights": {}},
                {"index": "4", "name": "recipient", "extra": "extra"},
            ]
        }
    }

    returned_value = _get_features("/file/name.gz")

    assert returned_value == expected_return_value


def test__get_gz_path():
    """Tests function ``_get_gz_path()``."""

    from dirty_cat.datasets.fetching import _get_gz_path

    # Passing invalid typed arguments
    with pytest.raises(ValueError):
        assert _get_gz_path(1, "", "")
        assert _get_gz_path("", 1, "")
        assert _get_gz_path("", "", 1)

    # Passing valid strings and verifying
    # it returns the correct expression.

    _platform = sys.platform
    if _platform.startswith("win"):
        # I wasn't able to make the test work on Windows...
        # I think it is os.path.join that decides for
        # some reason to remove the drive letter ("C:") every time...
        # e.g
        # AssertionError: assert 'C:\\path\\to\\file.gz' == '\\path\\to\\file.gz'
        # - C:\path\to\file.gz
        # ? --
        # + \path\to\file.gz
        return

    expected_return_value_1 = "/path/to/file.gz"
    expected_return_value_2 = "/tmp/path/to.strange/file.gz"
    expected_return_value_3 = "C:/Temp/path/to/very/very/very/very/long/file.gz"

    returned_value_1 = _get_gz_path("/", "path/to", "file")
    returned_value_2 = _get_gz_path("/tmp/", "/path/to.strange", "/file")
    returned_value_3 = _get_gz_path("C:/Temp/", "/path/to/very/very/very/very/long/", "file")

    assert returned_value_1 == expected_return_value_1
    assert returned_value_2 == expected_return_value_2
    assert returned_value_3 == expected_return_value_3


def test__export_gz_data_to_csv():
    """Tests function ``_export_gz_data_to_csv()``."""
    from dirty_cat.datasets.fetching import _export_gz_data_to_csv, Features

    features = Features(["top-left-square",
                         "top-middle-square",
                         "top-right-square",
                         "middle-left-square",
                         "middle-middle-square",
                         "middle-right-square",
                         "bottom-left-square",
                         "bottom-middle-square",
                         "bottom-right-square",
                         "Class"])
    arff_data = (
        "% This is a comment\n"
        "@relation tic-tac-toe\n"
        "@attribute 'top-left-square' {b,o,x}\n"
        "@data\n"
        "x,x,x,x,o,o,x,o,o,positive\n"
        "x,x,x,x,o,o,o,x,o,positive\n"
    )

    dummy_gz = "/dummy/file.gz"
    dummy_csv = "/dummy/file.csv"

    expected_calls = "[call('/dummy/file.csv', mode='w'),\n " \
                     "call().__enter__(),\n " \
                     "call().write('top-left-square,top-middle-square,top-right-square,middle-left-square," \
                     "middle-middle-square,middle-right-square,bottom-left-square,bottom-middle-square," \
                     "bottom-right-square,Class'),\n " \
                     "call().write('\\n'),\n " \
                     "call().write('x,x,x,x,o,o,x,o,o,positive\\n'),\n " \
                     "call().write('x,x,x,x,o,o,o,x,o,positive\\n'),\n " \
                     "call().__exit__(None, None, None)]"

    with mock.patch("builtins.open", mock_open(read_data="")) as mock_builtin_open:
        with mock.patch("gzip.open", mock_open(read_data=arff_data)) as mock_gzip_open:
            _export_gz_data_to_csv(dummy_gz, dummy_csv, features)
            mock_builtin_open   .assert_called_with(dummy_csv, mode='w')
            mock_gzip_open      .assert_called_with(dummy_gz, mode='rt')
            assert str(mock_builtin_open.mock_calls) == expected_calls


def test__features_to_csv_format():
    """Tests function ``_features_to_csv_format()``."""

    from dirty_cat.datasets.fetching import Features, _features_to_csv_format

    features = Features(["id", "name", "transaction_id", "owner", "recipient"])
    expected_return_value = "id,name,transaction_id,owner,recipient"
    assert _features_to_csv_format(features) == expected_return_value


@mock.patch("dirty_cat.datasets.fetching.fetch_openml_dataset")
def test_import_all_datasets(mock_fetch_openml_dataset):
    """Tests functions ``fetch_*()``."""

    from dirty_cat.datasets import fetching

    expected_return_value = {
        "description": "This is a dataset.",
        "source": "https://www.openml.org/",
        "path": "/path/to/file.csv",
    }
    mock_fetch_openml_dataset.return_value = expected_return_value

    returned_value = fetching.fetch_employee_salaries()
    mock_fetch_openml_dataset.assert_called_once_with(dataset_id=fetching.EMPLOYEE_SALARIES_ID)
    assert expected_return_value == returned_value

    mock_fetch_openml_dataset.reset_mock()

    returned_value = fetching.fetch_road_safety()
    mock_fetch_openml_dataset.assert_called_once_with(dataset_id=fetching.ROAD_SAFETY_ID)
    assert expected_return_value == returned_value

    mock_fetch_openml_dataset.reset_mock()

    returned_value = fetching.fetch_medical_charge()
    mock_fetch_openml_dataset.assert_called_once_with(dataset_id=fetching.MEDICAL_CHARGE_ID)
    assert expected_return_value == returned_value

    mock_fetch_openml_dataset.reset_mock()

    returned_value = fetching.fetch_midwest_survey()
    mock_fetch_openml_dataset.assert_called_once_with(dataset_id=fetching.MIDWEST_SURVEY_ID)
    assert expected_return_value == returned_value

    mock_fetch_openml_dataset.reset_mock()

    returned_value = fetching.fetch_open_payments()
    mock_fetch_openml_dataset.assert_called_once_with(dataset_id=fetching.OPEN_PAYMENTS_ID)
    assert expected_return_value == returned_value

    mock_fetch_openml_dataset.reset_mock()

    returned_value = fetching.fetch_traffic_violations()
    mock_fetch_openml_dataset.assert_called_once_with(dataset_id=fetching.TRAFFIC_VIOLATIONS_ID)
    assert expected_return_value == returned_value

    mock_fetch_openml_dataset.reset_mock()

    returned_value = fetching.fetch_drug_directory()
    mock_fetch_openml_dataset.assert_called_once_with(dataset_id=fetching.DRUG_DIRECTORY_ID)
    assert expected_return_value == returned_value
