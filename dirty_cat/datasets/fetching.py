# -*- coding: utf-8 -*-

"""
Fetching functions to retrieve example datasets, using
Scikit-Learn's ``fetch_openml()`` function.
"""

# Author: Lilian Boulard <lilian.boulard@inria.fr> || https://github.com/Phaide

import os
import gzip
import json
import warnings

from collections import namedtuple

from dirty_cat.datasets.utils import get_data_dir


Details = namedtuple("Details", ["name", "file_id", "description"])
Features = namedtuple("Features", ["names"])

# Directory where the ``.gz`` files containing the
# details on downloaded datasets are stored.
# Note: the tree structure is created by ``fetch_openml()``.
# As of october 2020, this function is annotated as
# ``Experimental`` so the structure might change in future releases.
# This path will be concatenated to the dirty_cat data directory,
# available via the function ``get_data_dir()``.
DETAILS_DIRECTORY = "openml/openml.org/api/v1/json/data/"

# Same as above ; for the datasets features location.
FEATURES_DIRECTORY = "openml/openml.org/api/v1/json/data/features/"

# Same as above ; for the datasets data location.
DATA_DIRECTORY = "openml/openml.org/data/v1/download/"

# The IDs of the datasets, from OpenML.
# This ID can be found in the URL.
# It is constructed as follows:
# https://www.openml.org/d/<ID>
ROAD_SAFETY_ID = 0
OPEN_PAYMENTS_ID = 0
MIDWEST_SURVEY_ID = 0
MEDICAL_CHARGE_ID = 0
EMPLOYEE_SALARIES_ID = 42125
TRAFFIC_VIOLATIONS_ID = 42132
DRUG_DIRECTORY_ID = 0


def fetch_openml_dataset(dataset_id: int, data_directory: str = get_data_dir()) -> dict:
    """
    Gets a dataset from OpenML (https://www.openml.org),
    or from the disk if already downloaded.

    Parameters
    ----------
    dataset_id: int
        The ID of the dataset to fetch.
    data_directory: str
        Optional. A directory to save the data to.
        By default, the dirty_cat data directory.

    Returns
    -------
    dict
        A dictionary containing:
          - ``description``
              The description of the dataset,
              as gathered from OpenML.
          - ``source``
              The dataset's URL from OpenML.
          - ``path``
              The absolute local path leading
              to the dataset (saved as a CSV file).

    """

    # Construct the path to the ``.gz`` file containing the details on a dataset.
    details_gz_path = _get_gz_path(data_directory, DETAILS_DIRECTORY, str(dataset_id))
    features_gz_path = _get_gz_path(data_directory, FEATURES_DIRECTORY, str(dataset_id))

    if not os.path.isfile(details_gz_path) or not os.path.isfile(features_gz_path):
        # If the details file or the features file
        # does not exist, download the dataset.
        warnings.warn(
            "Could not find the dataset locally. Downloading it from OpenML... This might take a while."
            "If the process is interrupted, some files will be invalid/incomplete."
            "To fix this problem, delete the CSV file if it exists. The system will recreate it on the next run."
        )
        _download_and_write_openml_dataset(dataset_id=dataset_id, data_directory=data_directory)
    details = _get_details(details_gz_path)

    # The file ID is required because the data file is named
    # after this ID, and not after the dataset's.
    file_id = details.file_id
    csv_path = os.path.join(data_directory, details.name + ".csv")

    data_gz_path = _get_gz_path(data_directory, DATA_DIRECTORY, str(file_id))

    if not os.path.isfile(data_gz_path):
        # This is a double-check.
        # If the data file does not exist, download the dataset.
        _download_and_write_openml_dataset(dataset_id=dataset_id, data_directory=data_directory)

    if not os.path.isfile(csv_path):
        # If the CSV file does not exist, use the dataset
        # downloaded by ``fetch_openml()`` to construct it.
        features = _get_features(features_gz_path)
        _export_gz_data_to_csv(data_gz_path, csv_path, features)

    url = "https://www.openml.org/d/{}".format(dataset_id)

    return {
        "description": details.description,
        "source": url,
        "path": csv_path
    }


def _download_and_write_openml_dataset(dataset_id: int, data_directory: str) -> None:
    """
    Downloads a dataset from OpenML,
    taking care of creating the tree structure.

    Parameters
    ----------
    dataset_id: int
        The ID of the dataset to download.
    data_directory: str
        The directory in which the data will be saved.

    """
    from sklearn.datasets import fetch_openml

    # The ``fetch_openml()`` function returns a Scikit-Learn ``Bunch`` object,
    # which behaves just like a ``namedtuple``.
    # However, we do not want to save this data into memory:
    # we will read it from the disk later.
    #
    # Raises ``ValueError`` if the ID is incorrect (does not exist on OpenML)
    # and ``urllib.error.URLError`` if there is no internet connection.
    fetch_openml(data_id=dataset_id, data_home=data_directory)


def _read_json_from_gz(compressed_dir_path: str) -> dict:
    """
    Opens a ``.gz`` file, reads its content
    (it expects JSON) and returns a dictionary.

    Parameters
    ----------
    compressed_dir_path
        Path to the ``.gz`` file to read.

    Returns
    -------
    dict
        The information contained in the file,
        converted from plain-text JSON.

    """
    if not os.path.isfile(compressed_dir_path):
        raise FileNotFoundError('Could not find file {}.'.format(compressed_dir_path))

    with gzip.open(compressed_dir_path, mode='rt') as gz:
        content = gz.read()

    details_json = json.JSONDecoder().decode(content)
    return details_json


def _get_details(compressed_dir_path: str) -> Details:
    """
    Gets useful details from the details file.

    Parameters
    ----------
    compressed_dir_path: str
        The path to the ``.gz`` file
        containing the details.

    Returns
    -------
    Details
        A ``Details`` object.

    """
    details = _read_json_from_gz(compressed_dir_path)["data_set_description"]
    # We filter out the irrelevant information.
    # If you want to modify this list (to add or remove items)
    # you must also modify the ``Details`` object definition.
    f_details = {
        "name": details["name"],
        "file_id": details["file_id"],
        "description": details["description"],
    }
    return Details(*f_details.values())


def _get_features(compressed_dir_path: str) -> Features:
    """
    Gets features that can be inserted in the CSV file
    or that can be useful in other ways.
    The most important feature being the column name.

    Parameters
    ----------
    compressed_dir_path
        Path to the ``.gz`` file
        containing the features.

    Returns
    -------
    Features
        A ``Features`` object.

    """
    raw_features = _read_json_from_gz(compressed_dir_path)["data_features"]
    # We filter out the irrelevant information.
    # If you want to modify this list (to add or remove items)
    # you must also modify the ``Details`` object definition.
    features = {
        "names": [column["name"] for column in raw_features["feature"]]
    }
    return Features(*features.values())


def _get_gz_path(root: str, directory: str, file_name: str) -> str:
    """
    Constructs the path to a ``.gz`` file.

    Parameters
    ----------
    root
        A directory tree starting from the system root,
        therefore, it must be absolute.
    directory
        Directory tree under root.
    file_name
        The file name, without the extension,
        and without any separator (such as "/" or "\")/

    Returns
    -------
    str
        The path to the compressed directory.

    """
    if not isinstance(root, str) or not isinstance(directory, str) or not isinstance(file_name, str):
        raise ValueError

    return os.path.join(
        root,
        directory.strip(os.sep),
        "{}.gz".format(file_name).strip(os.sep)
    )


def _export_gz_data_to_csv(compressed_dir_path: str, destination_file: str, features: Features) -> None:
    """
    Reads a ``.gz`` file containing an ARFF file,
    and writes to a target CSV the data.

    Parameters
    ----------
    compressed_dir_path: str
        Path to the ``.gz`` file containing the ARFF data.
    destination_file: str
        A CSV file to write to.
    features: Features
        A ``Features`` object containing the first CSV line
        (the columns' name).

    """
    atdata_found = False
    with open(destination_file, mode="w") as csv:
        with gzip.open(compressed_dir_path, mode="rt") as gz:
            csv.write(_features_to_csv_format(features))
            csv.write("\n")
            # We will look at each line of the file until we find
            # "@data": only after this tag is the actual CSV data.
            for line in gz:
                if not atdata_found:
                    if line.lower().startswith("@data"):
                        atdata_found = True
                else:
                    csv.write(line)


def _features_to_csv_format(features: Features) -> str:
    return ",".join(features.names)


# Datasets fetchers section


def fetch_employee_salaries() -> dict:
    """Fetches the employee_salaries dataset."""
    return fetch_openml_dataset(dataset_id=EMPLOYEE_SALARIES_ID)


def fetch_road_safety() -> dict:
    """Fetches the road safety dataset."""
    return fetch_openml_dataset(dataset_id=ROAD_SAFETY_ID)


def fetch_medical_charge() -> dict:
    """Fetches the medical charge dataset."""
    return fetch_openml_dataset(dataset_id=MEDICAL_CHARGE_ID)


def fetch_midwest_survey() -> dict:
    """Fetches the midwest survey dataset."""
    return fetch_openml_dataset(dataset_id=MIDWEST_SURVEY_ID)


def fetch_open_payments() -> dict:
    """Fetches the open payments dataset."""
    return fetch_openml_dataset(dataset_id=OPEN_PAYMENTS_ID)


def fetch_traffic_violations() -> dict:
    """Fetches the traffic violations dataset."""
    return fetch_openml_dataset(dataset_id=TRAFFIC_VIOLATIONS_ID)


def fetch_drug_directory() -> dict:
    """Fetches the drug directory dataset."""
    return fetch_openml_dataset(dataset_id=DRUG_DIRECTORY_ID)
