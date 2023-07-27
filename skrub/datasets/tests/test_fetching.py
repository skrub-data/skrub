import re
from tempfile import TemporaryDirectory
from unittest import mock
from urllib.error import URLError

import pandas as pd
import pytest

from skrub.datasets import _fetching


def _has_data_id(call, data_id: int) -> bool:
    # Unpacking copied from `mock._Call.__eq__`
    if len(call) == 2:
        args, kwargs = call
    else:
        name, args, kwargs = call
    return kwargs["data_id"] == data_id


@mock.patch(
    "skrub.datasets._fetching.fetch_openml",
    side_effect=_fetching.fetch_openml,
)
def test_openml_fetching(fetch_openml_mock: mock.Mock):
    """
    Downloads a small dataset (midwest survey) and performs a bunch of tests
    that asserts the fetching function works correctly.

    We don't download all datasets as it would take way too long:
    see https://github.com/skrub-data/skrub/issues/523
    """
    with TemporaryDirectory() as temp_dir:
        # Download the dataset without loading it in memory.
        dataset = _fetching.fetch_midwest_survey(
            data_directory=temp_dir,
            load_dataframe=False,
        )
        fetch_openml_mock.assert_called_once()
        assert _has_data_id(fetch_openml_mock.call_args, _fetching.MIDWEST_SURVEY_ID)
        fetch_openml_mock.reset_mock()
        assert isinstance(dataset, _fetching.DatasetInfoOnly)
        # Briefly load the dataframe in memory, to test that reading
        # from disk works.
        assert pd.read_csv(dataset.path, **dataset.read_csv_kwargs).shape == (2494, 29)
        assert dataset.name == (
            _fetching.fetch_midwest_survey.__name__[len("fetch_") :]
            .replace("_", " ")
            .capitalize()
        )
        assert dataset.source.startswith("https://www.openml.org/")
        assert str(_fetching.MIDWEST_SURVEY_ID) in dataset.source
        # Now, load it into memory, and expect `fetch_openml`
        # to not be called because the dataset is already on disk.
        dataset = _fetching.fetch_midwest_survey(data_directory=temp_dir)
        fetch_openml_mock.assert_not_called()
        fetch_openml_mock.reset_mock()
        assert dataset.X.shape == (2494, 28)
        assert dataset.y.shape == (2494,)


def test_openml_datasets_exist():
    """
    Queries OpenML to see if the datasets are still available on the website.
    """
    openml = pytest.importorskip("openml")
    openml.datasets.check_datasets_active(
        dataset_ids=[
            _fetching.ROAD_SAFETY_ID,
            _fetching.OPEN_PAYMENTS_ID,
            _fetching.MIDWEST_SURVEY_ID,
            _fetching.MEDICAL_CHARGE_ID,
            _fetching.EMPLOYEE_SALARIES_ID,
            _fetching.TRAFFIC_VIOLATIONS_ID,
            _fetching.DRUG_DIRECTORY_ID,
        ],
        raise_error_if_not_exist=True,
    )


@mock.patch("skrub.datasets._fetching.fetch_openml")
def test_openml_datasets_calls(fetch_openml_mock: mock.Mock):
    """
    Checks that calling the fetching functions actually calls
    `sklearn.datasets.fetch_openml`.
    Complementary to `test_openml_fetching`
    """
    with TemporaryDirectory() as temp_dir:
        for fetching_function, identifier in [
            (_fetching.fetch_road_safety, _fetching.ROAD_SAFETY_ID),
            (_fetching.fetch_open_payments, _fetching.OPEN_PAYMENTS_ID),
            (_fetching.fetch_midwest_survey, _fetching.MIDWEST_SURVEY_ID),
            (_fetching.fetch_medical_charge, _fetching.MEDICAL_CHARGE_ID),
            (_fetching.fetch_employee_salaries, _fetching.EMPLOYEE_SALARIES_ID),
            (_fetching.fetch_traffic_violations, _fetching.TRAFFIC_VIOLATIONS_ID),
            (_fetching.fetch_drug_directory, _fetching.DRUG_DIRECTORY_ID),
        ]:
            try:
                fetching_function(data_directory=temp_dir)
            except FileNotFoundError:
                pass
            fetch_openml_mock.assert_called_once()
            assert _has_data_id(fetch_openml_mock.call_args, identifier)
            fetch_openml_mock.reset_mock()


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

    with TemporaryDirectory() as temp_dir:
        try:
            # First, we want to purposefully test FileNotFoundError exceptions.
            with pytest.raises(FileNotFoundError):
                assert _fetching.fetch_world_bank_indicator(
                    indicator_id=0, data_directory=temp_dir
                )
                assert _fetching.fetch_world_bank_indicator(
                    indicator_id=2**32,
                    data_directory=temp_dir,
                )

            # Valid call
            returned_info = _fetching.fetch_world_bank_indicator(
                indicator_id=test_dataset["id"],
                data_directory=temp_dir,
            )

        except (ConnectionError, URLError):
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
            disk_loaded_info = _fetching.fetch_world_bank_indicator(
                indicator_id=test_dataset["id"],
                data_directory=temp_dir,
            )
            mock_urlretrieve.assert_not_called()
            assert disk_loaded_info == returned_info


def test_fetch_movielens():
    """
    Tests the ``fetch_movielens()`` function in a real environment.
    """
    test_dataset = {
        "id": "ratings",
        "desc_start": "Summary\n=======\n\nThis dataset (ml-latest-small)",
        "url": "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip",  # noqa
        "dataset_columns_count": 4,
    }

    with TemporaryDirectory() as temp_dir:
        try:
            # First, we want to purposefully test ValueError exceptions.
            msg = "dataset_id options are ['movies', 'ratings'], got 'wrong_name'."
            with pytest.raises(ValueError, match=re.escape(msg)):
                assert _fetching.fetch_movielens(
                    dataset_id="wrong_name", data_directory=temp_dir
                )

            # Valid call
            returned_info = _fetching.fetch_movielens(
                dataset_id="ratings",
                data_directory=temp_dir,
            )

        except (ConnectionError, URLError):
            pytest.skip(
                "Exception: Skipping this test because we encountered an "
                "issue probably related to an Internet connection problem. "
            )
            return

        assert returned_info.description.startswith(test_dataset["desc_start"])
        assert returned_info.source == test_dataset["url"]
        assert returned_info.path.is_file()

        dataset = pd.read_csv(returned_info.path)

        assert dataset.columns[0] == "userId"
        assert dataset.shape[1] == test_dataset["dataset_columns_count"]

        # Now that we have verified the file is on disk, we want to test
        # whether calling the function again reads it from disk (it should)
        # or queries the network again (it shouldn't).
        with mock.patch("urllib.request.urlretrieve") as mock_urlretrieve:
            # Same valid call as above
            disk_loaded_info = _fetching.fetch_movielens(
                dataset_id=test_dataset["id"],
                data_directory=temp_dir,
            )
            mock_urlretrieve.assert_not_called()
            assert disk_loaded_info == returned_info
