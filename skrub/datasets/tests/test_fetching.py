from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock
from urllib.error import URLError

import pytest
from pandas.testing import assert_frame_equal

from skrub.datasets import fetch_midwest_survey, fetch_world_bank_indicator
from skrub.datasets._fetching_functions import (
    DRUG_DIRECTORY_ID,
    EMPLOYEE_SALARIES_ID,
    MEDICAL_CHARGE_ID,
    MIDWEST_SURVEY_ID,
    OPEN_PAYMENTS_ID,
    ROAD_SAFETY_ID,
    TRAFFIC_VIOLATIONS_ID,
)


def test_openml_fetching() -> None:
    """
    Downloads a small dataset (midwest survey) and performs a few tests
    that asserts the fetching function works correctly.

    We don't download all datasets as it would take way too long:
    see https://github.com/skrub-data/skrub/issues/523
    """
    with TemporaryDirectory() as temp_dir:
        dataset = fetch_midwest_survey(directory=temp_dir)
        assert dataset.X.shape == (2494, 27)
        assert dataset.y.shape == (2494,)
        # Assert there is at least one file named after the dataset ID
        # in the temporary directory tree.
        assert Path(temp_dir).rglob(f"*{MIDWEST_SURVEY_ID}*")


def test_openml_datasets_exist() -> None:
    """
    Queries OpenML to see if the datasets are still available on the website.
    """
    openml = pytest.importorskip("openml")
    openml.datasets.check_datasets_active(
        dataset_ids=[
            ROAD_SAFETY_ID,
            OPEN_PAYMENTS_ID,
            MIDWEST_SURVEY_ID,
            MEDICAL_CHARGE_ID,
            EMPLOYEE_SALARIES_ID,
            TRAFFIC_VIOLATIONS_ID,
            DRUG_DIRECTORY_ID,
        ],
        raise_error_if_not_exist=True,
    )


def test_fetch_world_bank_indicator() -> None:
    """
    Tests ``fetch_world_bank_indicator()`` in a real environment.
    """
    test_dataset = {
        "id": "NY.GDP.PCAP.CD",
        "desc_start": "This table shows",
        "url": (
            "https://api.worldbank.org/v2/en"
            "/indicator/NY.GDP.PCAP.CD?downloadformat=csv"
        ),
        "dataset_columns_count": 2,
    }

    with TemporaryDirectory() as temp_dir:
        try:
            # First, we want to purposefully test FileNotFoundError exceptions.
            with pytest.raises(FileNotFoundError):
                assert fetch_world_bank_indicator(
                    indicator_id="blablabla", directory=temp_dir
                )
                assert fetch_world_bank_indicator(
                    indicator_id="I don't exist",
                    directory=temp_dir,
                )

            # Valid call
            dataset = fetch_world_bank_indicator(
                indicator_id=test_dataset["id"],
                directory=temp_dir,
            )

        except (ConnectionError, URLError):
            pytest.skip(
                "Exception: Skipping this test because we encountered an "
                "issue probably related to an Internet connection problem. "
            )

        assert dataset.description.startswith(test_dataset["desc_start"])
        assert dataset.source == test_dataset["url"]
        assert dataset.X.columns[0] == "Country Name"
        assert dataset.X.shape[1] == test_dataset["dataset_columns_count"]

        # Now that we have verified the file is on disk, we want to test
        # whether calling the function again reads it from disk (it should)
        # or queries the network again (it shouldn't).
        with mock.patch("urllib.request.urlretrieve") as mock_urlretrieve:
            # Same valid call as above
            dataset_local = fetch_world_bank_indicator(
                indicator_id=test_dataset["id"],
                directory=temp_dir,
            )
            mock_urlretrieve.assert_not_called()
            assert_frame_equal(dataset.X, dataset_local.X)
