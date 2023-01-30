"""
Tests fetching.py (datasets fetching off OpenML.org).
"""

import shutil
import warnings
from functools import wraps
from json import JSONDecodeError
from pathlib import Path
from unittest import mock
from unittest.mock import mock_open
from urllib.error import URLError

import pandas as pd
import pytest

from dirty_cat.datasets import _fetching
from dirty_cat.datasets._fetching import (
    Details,
    Features,
    _download_and_write_openml_dataset,
    _export_gz_data_to_csv,
    _features_to_csv_format,
    _fetch_openml_dataset,
    _get_details,
    _get_features,
    _read_json_from_gz,
)
from dirty_cat.datasets._fetching import (
    fetch_world_bank_indicator as fetch_world_bank_indicator,
)
from dirty_cat.datasets._utils import get_data_dir as _get_data_dir


@wraps(_fetch_openml_dataset)
def fetch_openml_dataset(*args, **kwargs):
    """
    Wrapper for the fetching function.
    Filters out specific warnings.
    """
    with warnings.catch_warnings():
        warnings.simplefilter(
            action="ignore",
            category=UserWarning,
        )
        return _fetch_openml_dataset(*args, **kwargs)


def get_test_data_dir() -> Path:
    return _get_data_dir("tests")


def test_fetch_openml_dataset():
    """
    Tests the ``_fetch_openml_dataset()`` function in a real environment.
    Though, to avoid the test being too long,
    we will download a small dataset (<1000 entries).

    Reference:
    https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/tests/test_openml.py
    """

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
                assert fetch_openml_dataset(dataset_id=0, data_directory=test_data_dir)
                assert fetch_openml_dataset(
                    dataset_id=2**32, data_directory=test_data_dir
                )

            # Valid call
            returned_info = fetch_openml_dataset(
                dataset_id=test_dataset["id"],
                data_directory=test_data_dir,
            )

        except URLError:
            warnings.warn(
                "No internet connection or the website is down, test aborted."
            )
            # One could try to manually recreate the tree structure
            # created by ``fetch_openml()``,  and the
            # ``.gz`` files within in order to finish the test.
            pytest.skip(
                "Exception: Skipping this test because we encountered an "
                "issue probably related to an Internet connection problem. "
            )
            return

        assert returned_info["description"].startswith(test_dataset["desc_start"])
        assert returned_info["source"] == test_dataset["url"]
        assert returned_info["path"] == test_data_dir / test_dataset["csv_name"]

        assert returned_info["path"].is_file()

        dataset = pd.read_csv(
            returned_info["path"], sep=",", quotechar="'", escapechar="\\"
        )

        assert dataset.shape == (
            test_dataset["dataset_rows_count"],
            test_dataset["dataset_columns_count"],
        )

        # Now that we have verified the file is on disk, we want to test
        # whether calling the function again reads it from disk (it should)
        # or queries the network again (it shouldn't).
        with mock.patch("sklearn.datasets.fetch_openml") as mock_fetch_openml:
            # Same valid call as above
            disk_loaded_info = fetch_openml_dataset(
                dataset_id=test_dataset["id"],
                data_directory=test_data_dir,
            )
            mock_fetch_openml.assert_not_called()
            assert disk_loaded_info == returned_info
    finally:
        shutil.rmtree(path=str(test_data_dir), ignore_errors=True)


@mock.patch("dirty_cat.datasets._fetching.Path.is_file")
@mock.patch("dirty_cat.datasets._fetching._get_features")
@mock.patch("dirty_cat.datasets._fetching._get_details")
@mock.patch("dirty_cat.datasets._fetching._export_gz_data_to_csv")
@mock.patch("dirty_cat.datasets._fetching._download_and_write_openml_dataset")
def test_fetch_openml_dataset_mocked(
    mock_download,
    mock_export,
    mock_get_details,
    mock_get_features,
    mock_pathlib_path_isfile,
):
    """
    Test function ``_fetch_openml_dataset()``, but this time,
    we mock the functions to test its inner mechanisms.
    """

    mock_get_details.return_value = Details(
        name="Dataset_name", file_id="123456", description="Dummy dataset description."
    )
    mock_get_features.return_value = Features(
        names=["id", "name", "transaction_id", "owner", "recipient"]
    )

    test_data_dir = get_test_data_dir()

    # We test the function to see if it behaves correctly when files exist
    mock_pathlib_path_isfile.return_value = True

    fetch_openml_dataset(50, test_data_dir)

    mock_download.assert_not_called()
    mock_export.assert_not_called()
    mock_get_features.assert_not_called()
    mock_get_details.assert_called_once()

    # Reset mocks
    mock_get_details.reset_mock()

    # This time, the files do not exist
    mock_pathlib_path_isfile.return_value = False

    fetch_openml_dataset(50, test_data_dir)

    # Download should be called twice
    mock_download.assert_called_with(dataset_id=50, data_directory=get_test_data_dir())
    mock_export.assert_called_once()
    mock_get_features.assert_called_once()
    mock_get_details.assert_called_once()


@mock.patch("dirty_cat.datasets._fetching.fetch_openml")
def test__download_and_write_openml_dataset(mock_fetch_openml):
    """Tests function ``_download_and_write_openml_dataset()``."""

    dataset_id = 2

    test_data_dir = get_test_data_dir()
    _download_and_write_openml_dataset(dataset_id, test_data_dir)

    mock_fetch_openml.assert_called_once_with(
        data_id=dataset_id, data_home=str(test_data_dir), as_frame=True
    )


@mock.patch("dirty_cat.datasets._fetching.Path.is_file")
def test__read_json_from_gz(mock_pathlib_path_isfile):
    """Tests function ``_read_json_from_gz()``."""

    dummy_file_path = Path("file/path.gz")

    # Passing an invalid file path (does not exist).
    mock_pathlib_path_isfile.return_value = False
    with pytest.raises(FileNotFoundError):
        assert _read_json_from_gz(dummy_file_path)

    # Passing a valid file path,
    # but reading it does not return JSON-encoded data.
    mock_pathlib_path_isfile.return_value = True
    with mock.patch(
        "gzip.open", mock_open(read_data="This is not JSON-encoded data!")
    ) as _:
        with pytest.raises(JSONDecodeError):
            assert _read_json_from_gz(dummy_file_path)

    # Passing a valid file path, and reading it
    # returns valid JSON-encoded data.
    expected_return_value = {"data": "This is JSON-encoded data!"}
    with mock.patch(
        "gzip.open", mock_open(read_data='{"data": "This is JSON-encoded data!"}')
    ) as _:
        assert _read_json_from_gz(dummy_file_path) == expected_return_value


@mock.patch("dirty_cat.datasets._fetching._read_json_from_gz")
def test__get_details(mock_read_json_from_gz):
    """Tests function ``_get_details()``."""

    expected_return_value = Details(
        "Dataset_name", "123456", "Dummy dataset description."
    )

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


@mock.patch("dirty_cat.datasets._fetching._read_json_from_gz")
def test__get_features(mock_read_json_from_gz):
    """Tests function ``_get_features()``."""

    expected_return_value = Features(
        ["id", "name", "transaction_id", "owner", "recipient"]
    )

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

    features = Features(
        [
            "top-left-square",
            "top-middle-square",
            "top-right-square",
            "middle-left-square",
            "middle-middle-square",
            "middle-right-square",
            "bottom-left-square",
            "bottom-middle-square",
            "bottom-right-square",
            "Class",
        ]
    )
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

    with mock.patch(
        "pathlib.Path.open", mock_open(read_data="")
    ) as mock_pathlib_path_open:
        with mock.patch("gzip.open", mock_open(read_data=arff_data)) as mock_gzip_open:
            _export_gz_data_to_csv(dummy_gz, dummy_csv, features)
            mock_pathlib_path_open.assert_called_with(mode="w", encoding="utf8")
            mock_gzip_open.assert_called_with(dummy_gz, mode="rt", encoding="utf8")


def test__features_to_csv_format():
    """Tests function ``_features_to_csv_format()``."""

    features = Features(["id", "name", "transaction_id", "owner", "recipient"])
    expected_return_value = "id,name,transaction_id,owner,recipient"
    assert _features_to_csv_format(features) == expected_return_value


@mock.patch("dirty_cat.datasets._fetching._fetch_openml_dataset")
@mock.patch("dirty_cat.datasets._fetching._fetch_dataset_as_dataclass")
def test_import_all_datasets(
    mock_fetch_dataset_as_dataclass, mock_fetch_openml_dataset
):
    """Tests functions ``fetch_*()``."""

    mock_fetch_openml_dataset.return_value = {
        "name": "Example dataset",
        "description": "This is a dataset.",
        "source": "https://www.openml.org/",
        "target": "To_predict",
        "path": Path("/path/to/file.csv"),
        "read_csv_kwargs": {"a": "b"},
    }

    expected_return_value_all = _fetching.DatasetAll(
        name="Example dataset",
        description="This is a dataset.",
        source="https://www.openml.org/",
        target="To_predict",
        path=Path("/path/to/file.csv"),
        X=pd.DataFrame([1, 2, 3, 4]),
        y=pd.Series([5]),
        read_csv_kwargs={"a": "b"},
    )

    expected_return_value_info_only = _fetching.DatasetInfoOnly(
        name="Example dataset",
        description="This is a dataset.",
        source="https://www.openml.org/",
        target="To_predict",
        path=Path("/path/to/file.csv"),
        read_csv_kwargs={"a": "b"},
    )

    for expected_return_value, load_dataframe in zip(
        [expected_return_value_all, expected_return_value_info_only],
        [True, False],
    ):
        mock_fetch_dataset_as_dataclass.return_value = expected_return_value

        returned_value = _fetching.fetch_employee_salaries(
            drop_linked=False, drop_irrelevant=False
        )
        assert expected_return_value == returned_value

        mock_fetch_openml_dataset.reset_mock()

        returned_value = _fetching.fetch_road_safety()
        assert expected_return_value == returned_value

        mock_fetch_openml_dataset.reset_mock()

        returned_value = _fetching.fetch_medical_charge()
        assert expected_return_value == returned_value

        mock_fetch_openml_dataset.reset_mock()

        returned_value = _fetching.fetch_midwest_survey()
        assert expected_return_value == returned_value

        mock_fetch_openml_dataset.reset_mock()

        returned_value = _fetching.fetch_open_payments()
        assert expected_return_value == returned_value

        mock_fetch_openml_dataset.reset_mock()

        returned_value = _fetching.fetch_traffic_violations()
        assert expected_return_value == returned_value

        mock_fetch_openml_dataset.reset_mock()

        returned_value = _fetching.fetch_drug_directory()
        assert expected_return_value == returned_value


def test_fetch_world_bank_indicator():
    """
    Tests the ``fetch_world_bank_indicator()``
    function in a real environment.
    """
    test_dataset = {
        "id": "NY.GDP.PCAP.CD",
        "desc_start": "This table shows",
        "url": "https://api.worldbank.org/v2/en/indicator/NY.GDP.PCAP.CD?downloadformat=csv",  # noqa
        "dataset_columns_count": 2,
    }

    test_data_dir = get_test_data_dir()

    try:
        try:
            # First, we want to purposefully test FileNotFoundError exceptions.
            with pytest.raises(FileNotFoundError):
                assert fetch_world_bank_indicator(indicator_id=0)
                assert fetch_world_bank_indicator(indicator_id=2**32)

            # Valid call
            returned_info = fetch_world_bank_indicator(indicator_id=test_dataset["id"])

        except URLError:
            warnings.warn(
                "No internet connection or the website is down, test aborted."
            )
            pytest.skip(
                "Exception: Skipping this test because we encountered an "
                "issue probably related to an Internet connection problem. "
            )
            return

        assert returned_info.description.startswith(test_dataset["desc_start"])
        assert returned_info.source == test_dataset["url"]
        assert returned_info.path.is_file()

        dataset = pd.read_csv(returned_info.path)

        assert dataset.columns[0] == "Country Name"
        assert dataset.shape[1] == test_dataset["dataset_columns_count"]

        # Now that we have verified the file is on disk, we want to test
        # whether calling the function again reads it from disk (it should)
        # or queries the network again (it shouldn't).
        with mock.patch("urllib.request.urlretrieve") as mock_urlretrieve:
            # Same valid call as above
            disk_loaded_info = fetch_world_bank_indicator(
                indicator_id=test_dataset["id"]
            )
            mock_urlretrieve.assert_not_called()
            assert disk_loaded_info == returned_info

    finally:
        shutil.rmtree(path=str(test_data_dir), ignore_errors=True)


