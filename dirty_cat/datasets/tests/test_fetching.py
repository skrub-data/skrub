"""

Tests fetching.py (datasets fetching off OpenML.org).

"""

# Author: Lilian Boulard <lilian@boulard.fr> || https://github.com/LilianBoulard

import pytest
import shutil
import sklearn
import pandas as pd

from pathlib import Path
from distutils.version import LooseVersion

from unittest import mock
from unittest.mock import mock_open


def get_test_data_dir() -> Path:
    from dirty_cat.datasets.utils import get_data_dir
    return get_data_dir("tests")


def test_fetch_openml_dataset():
    """
    Tests the ``fetch_openml_dataset()`` function in a real environment.
    Though, to avoid the test being too long, we will download a small dataset (<1000 entries).

    Reference: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/tests/test_openml.py
    """

    from dirty_cat.datasets.fetching import fetch_openml_dataset
    from urllib.error import URLError

    # Hard-coded information about the test dataset.
    # Centralizes used information here.
    test_dataset = {
        "id": 50,
        "desc_start": "**Author**",
        "url": "https://www.openml.org/d/50",
        "csv_name": "tic-tac-toe.csv",
        "dataset_rows_count": 958,
        "dataset_columns_count": 10,
    }

    test_data_dir = get_test_data_dir()

    try:  # Try-finally block used to remove the test data directory at the end
        try:

            # First, we want to purposefully test ValueError exceptions.
            with pytest.raises(ValueError):
                assert fetch_openml_dataset(dataset_id=0,
                                            data_directory=test_data_dir)
                assert fetch_openml_dataset(dataset_id=2**32,
                                            data_directory=test_data_dir)

            # Valid call
            returned_info = fetch_openml_dataset(dataset_id=test_dataset["id"],
                                                 data_directory=test_data_dir)

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
        assert returned_info["path"] == test_data_dir / test_dataset["csv_name"]

        assert returned_info["path"].is_file()

        dataset: pd.DataFrame = pd.read_csv(returned_info["path"], sep=",",
                                            quotechar="'", escapechar="\\")

        assert dataset.shape == (test_dataset["dataset_rows_count"],
                                 test_dataset["dataset_columns_count"])

    finally:
        shutil.rmtree(path=str(test_data_dir), ignore_errors=True)


@mock.patch("pathlib.Path.is_file")
@mock.patch("dirty_cat.datasets.fetching._get_features")
@mock.patch("dirty_cat.datasets.fetching._get_details")
@mock.patch("dirty_cat.datasets.fetching._export_gz_data_to_csv")
@mock.patch("dirty_cat.datasets.fetching._download_and_write_openml_dataset")
def test_fetch_openml_dataset_mocked(mock_download, mock_export,
                                     mock_get_details, mock_get_features,
                                     mock_pathlib_path_isfile):
    """
    Test function ``fetch_openml_dataset()``, but this time,
    we mock the functions to test its inner mechanisms.
    """

    from dirty_cat.datasets.fetching import fetch_openml_dataset, Details, Features

    mock_get_details.return_value = Details("Dataset_name", "123456",
                                            "Dummy dataset description.")
    mock_get_features.return_value = Features(["id", "name", "transaction_id",
                                               "owner", "recipient"])

    test_data_dir = get_test_data_dir()

    # We test the function to see if it behaves correctly when files exists
    mock_pathlib_path_isfile.return_value = True

    fetch_openml_dataset(50, test_data_dir)

    mock_download       .assert_not_called()
    mock_export         .assert_not_called()
    mock_get_features   .assert_not_called()
    mock_get_details    .assert_called_once()

    # Reset mocks
    mock_get_details    .reset_mock()

    # This time, the files does not exists
    mock_pathlib_path_isfile.return_value = False

    fetch_openml_dataset(50, test_data_dir)

    # Download should be called twice
    mock_download       .assert_called_with(dataset_id=50,
                                            data_directory=get_test_data_dir())
    mock_export         .assert_called_once()
    mock_get_features   .assert_called_once()
    mock_get_details    .assert_called_once()


@mock.patch('sklearn.datasets.fetch_openml')
def test__download_and_write_openml_dataset(mock_fetch_openml):
    """Tests function ``_download_and_write_openml_dataset()``."""

    from dirty_cat.datasets.fetching import _download_and_write_openml_dataset

    test_data_dir = get_test_data_dir()
    _download_and_write_openml_dataset(1, test_data_dir)

    if LooseVersion(sklearn.__version__) >= LooseVersion('0.22'):
        mock_fetch_openml.assert_called_once_with(data_id=1,
                                                  data_home=str(test_data_dir),
                                                  as_frame=True)
    else:
        mock_fetch_openml.assert_called_once_with(data_id=1,
                                                  data_home=str(test_data_dir))


@mock.patch("pathlib.Path.is_file")
def test__read_json_from_gz(mock_pathlib_path_isfile):
    """Tests function ``_read_json_from_gz()``."""

    from dirty_cat.datasets.fetching import _read_json_from_gz
    from json import JSONDecodeError

    dummy_file_path = Path("file/path.gz")

    # Passing an invalid file path (does not exist).
    mock_pathlib_path_isfile.return_value = False
    with pytest.raises(FileNotFoundError):
        assert _read_json_from_gz(dummy_file_path)

    # Passing a valid file path,
    # but reading it does not return JSON-encoded data.
    mock_pathlib_path_isfile.return_value = True
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

    expected_return_value = Details("Dataset_name", "123456",
                                    "Dummy dataset description.")

    mock_read_json_from_gz.return_value = {
        "data_set_description": {
            "data": "This is JSON data!",
            "name": "Dataset_name",
            "file_id": "123456",
            "description": "Dummy dataset description.",
            "extra": "extra_field",
        }
    }

    returned_value = _get_details(Path("/file/name.gz"))

    assert returned_value == expected_return_value


@mock.patch("dirty_cat.datasets.fetching._read_json_from_gz")
def test__get_features(mock_read_json_from_gz):
    """Tests function ``_get_features()``."""

    from dirty_cat.datasets.fetching import Features, _get_features

    expected_return_value = Features(["id", "name", "transaction_id",
                                      "owner", "recipient"])

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

    returned_value = _get_features(Path("/file/name.gz"))

    assert returned_value == expected_return_value


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

    dummy_gz = Path("/dummy/file.gz")
    dummy_csv = Path("/dummy/file.csv")

    expected_calls = "[call(mode='w'),\n " \
                     "call().__enter__(),\n " \
                     "call().write('top-left-square,top-middle-square,top-right-square,middle-left-square," \
                     "middle-middle-square,middle-right-square,bottom-left-square,bottom-middle-square," \
                     "bottom-right-square,Class'),\n " \
                     "call().write('\\n'),\n " \
                     "call().write('x,x,x,x,o,o,x,o,o,positive\\n'),\n " \
                     "call().write('x,x,x,x,o,o,o,x,o,positive\\n'),\n " \
                     "call().__exit__(None, None, None)]"

    with mock.patch("pathlib.Path.open", mock_open(read_data="")) as mock_pathlib_path_open:
        with mock.patch("gzip.open", mock_open(read_data=arff_data)) as mock_gzip_open:
            _export_gz_data_to_csv(dummy_gz, dummy_csv, features)
            mock_pathlib_path_open.assert_called_with(mode='w')
            mock_gzip_open.assert_called_with(dummy_gz, mode='rt')
            assert str(mock_pathlib_path_open.mock_calls) == expected_calls


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
        "path": Path("/path/to/file.csv"),
    }
    mock_fetch_openml_dataset.return_value = expected_return_value

    returned_value = fetching.fetch_employee_salaries()
    mock_fetch_openml_dataset.assert_called_once_with(
        dataset_id=fetching.EMPLOYEE_SALARIES_ID)
    assert expected_return_value == returned_value

    mock_fetch_openml_dataset.reset_mock()

    returned_value = fetching.fetch_road_safety()
    mock_fetch_openml_dataset.assert_called_once_with(
        dataset_id=fetching.ROAD_SAFETY_ID)
    assert expected_return_value == returned_value

    mock_fetch_openml_dataset.reset_mock()

    returned_value = fetching.fetch_medical_charge()
    mock_fetch_openml_dataset.assert_called_once_with(
        dataset_id=fetching.MEDICAL_CHARGE_ID)
    assert expected_return_value == returned_value

    mock_fetch_openml_dataset.reset_mock()

    returned_value = fetching.fetch_midwest_survey()
    mock_fetch_openml_dataset.assert_called_once_with(
        dataset_id=fetching.MIDWEST_SURVEY_ID)
    assert expected_return_value == returned_value

    mock_fetch_openml_dataset.reset_mock()

    returned_value = fetching.fetch_open_payments()
    mock_fetch_openml_dataset.assert_called_once_with(
        dataset_id=fetching.OPEN_PAYMENTS_ID)
    assert expected_return_value == returned_value

    mock_fetch_openml_dataset.reset_mock()

    returned_value = fetching.fetch_traffic_violations()
    mock_fetch_openml_dataset.assert_called_once_with(
        dataset_id=fetching.TRAFFIC_VIOLATIONS_ID)
    assert expected_return_value == returned_value

    mock_fetch_openml_dataset.reset_mock()

    returned_value = fetching.fetch_drug_directory()
    mock_fetch_openml_dataset.assert_called_once_with(
        dataset_id=fetching.DRUG_DIRECTORY_ID)
    assert expected_return_value == returned_value


if __name__ == "__main__":
    print("Tests starting")
    test_fetch_openml_dataset_mocked()
    test_fetch_openml_dataset()
    print("Tests passed")
