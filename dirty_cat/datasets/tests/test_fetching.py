import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable
from unittest import mock
from urllib.error import URLError

import pandas as pd
import pytest

from dirty_cat.datasets import _fetching


@pytest.mark.parametrize(
    ("fetching_function",),
    [
        (_fetching.fetch_road_safety,),
        (_fetching.fetch_medical_charge,),
        (_fetching.fetch_midwest_survey,),
        (_fetching.fetch_open_payments,),
        (_fetching.fetch_traffic_violations,),
        (_fetching.fetch_drug_directory,),
        (_fetching.fetch_employee_salaries,),
    ],
)
def test_openml_fetching(fetching_function: Callable):
    """
    Test a function that loads data from OpenML.
    """
    with TemporaryDirectory() as temp_dir_1, TemporaryDirectory() as temp_dir_2:
        # Convert to path objects
        temp_dir_1 = Path(temp_dir_1).absolute()
        temp_dir_2 = Path(temp_dir_2).absolute()

        # Fetch without loading into memory
        try:
            dataset_wo_load: _fetching.DatasetInfoOnly = fetching_function(
                load_dataframe=False, directory=temp_dir_1
            )
        except (ConnectionError,) as e:
            pytest.skip(f"Got a network error ({e}), skipping.")
            return
        else:
            assert isinstance(dataset_wo_load, _fetching.DatasetInfoOnly)

        # FIXME: An more elegant way of testing whether the dataset is loaded
        #  in memory would be to monitor it and expect it to stay quite low.
        assert not hasattr(dataset_wo_load, "X")
        assert not hasattr(dataset_wo_load, "y")

        # Now that we've made the checks specific to the unloaded dataset,
        # we'll load it from disk and store it for later.
        from_disk_df = pd.read_csv(
            dataset_wo_load.path,
            **dataset_wo_load.read_csv_kwargs,
        )
        y = from_disk_df[dataset_wo_load.target]
        X = from_disk_df.drop(dataset_wo_load.target, axis="columns")
        # Essentially convert the DatasetInfoOnly to DatasetAll
        dataset_wo_load_loaded = _fetching.DatasetAll(
            name=dataset_wo_load.name,
            description=dataset_wo_load.description,
            source=dataset_wo_load.source,
            target=dataset_wo_load.target,
            X=X,
            y=y,
            path=dataset_wo_load.path,
            read_csv_kwargs=dataset_wo_load.read_csv_kwargs,
        )

        # Fetch and load into memory
        try:
            dataset_w_load: _fetching.DatasetAll = fetching_function(
                directory=temp_dir_2
            )
        except (ConnectionError,) as e:
            pytest.skip(f"Got a network error ({e}), skipping.")
            return
        else:
            assert isinstance(dataset_w_load, _fetching.DatasetAll)
            from_disk_df = pd.read_csv(
                dataset_w_load.path,
                **dataset_w_load.read_csv_kwargs,
            )
            y = from_disk_df[dataset_w_load.target]
            X = from_disk_df.drop(dataset_w_load.target, axis="columns")
            pd.testing.assert_frame_equal(X, dataset_w_load.X)
            pd.testing.assert_series_equal(y, dataset_w_load.y)

        # Execute standard checks for both type of gathered datasets
        for dataset in (dataset_w_load, dataset_wo_load_loaded):
            dataset: _fetching.DatasetAll

            assert dataset.path.exists()
            # Expect at least a few lines and columns
            assert dataset.X.shape > (5, 5)
            assert dataset.y.shape > (5,)
            assert dataset.X.shape[0] == dataset.y.shape[0]

            # Less important checks, but might help finding errors
            assert dataset.name == (
                fetching_function.__name__[len("fetch_") :]
                .replace("_", " ")
                .capitalize()
            )
            assert dataset.source.startswith("https://www.openml.org/")


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
        temp_dir = Path(temp_dir).absolute()
        try:
            # First, we want to purposefully test FileNotFoundError exceptions.
            with pytest.raises(FileNotFoundError):
                assert _fetching.fetch_world_bank_indicator(
                    indicator_id=0, directory=temp_dir
                )
                assert _fetching.fetch_world_bank_indicator(
                    indicator_id=2**32, directory=temp_dir
                )

            # Valid call
            returned_info = _fetching.fetch_world_bank_indicator(
                indicator_id=test_dataset["id"], directory=temp_dir
            )

        except (ConnectionError, URLError):
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
            disk_loaded_info = _fetching.fetch_world_bank_indicator(
                indicator_id=test_dataset["id"]
            )
            mock_urlretrieve.assert_not_called()
            assert disk_loaded_info == returned_info
