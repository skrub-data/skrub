import hashlib
import json
import os
import shutil
import tempfile
import time
import warnings
from pathlib import Path

import pandas as pd
import requests
from sklearn.utils import Bunch

DATA_HOME_ENVAR_NAME = "SKRUB_DATA_DIRECTORY"
DATASET_INFO = {
    "bike_sharing": {
        "urls": [
            "https://github.com/skrub-data/skrub-data-files/raw/refs/heads/main/bike_sharing.zip",
            "https://osf.io/download/3z4qc",
        ],
        "sha256": "33745414801712034cf1d8615d7f086bba411ea8e44bfffefc0c6f23cb8afb83",
    },
    "country_happiness": {
        "urls": [
            "https://github.com/skrub-data/skrub-data-files/raw/refs/heads/main/country_happiness.zip",
            "https://osf.io/download/8a6wm",
        ],
        "sha256": "10b35da781a13a94dedcfeb43b291d16677b06a781e9b88a780f04ad173b422d",
    },
    "credit_fraud": {
        "urls": [
            "https://github.com/skrub-data/skrub-data-files/raw/refs/heads/main/credit_fraud.zip",
            "https://osf.io/download/y8qg5",
        ],
        "sha256": "ec40d370a275d4bd2637d4c617120e91e2e7946d23c93b1a1ea7df824ee1e514",
    },
    "drug_directory": {
        "urls": [
            "https://github.com/skrub-data/skrub-data-files/raw/refs/heads/main/drug_directory.zip",
            "https://osf.io/download/rtgk5",
        ],
        "sha256": "0c3885894baf02fc787109801ec2c34cc25cd4a31e0066a16941b74157474887",
    },
    "employee_salaries": {
        "urls": [
            "https://github.com/skrub-data/skrub-data-files/raw/refs/heads/main/employee_salaries.zip",
            "https://osf.io/download/c592f",
        ],
        "sha256": "4b4919f38d921014cb1fd24ad302f44bccc55d1eeeeb8482902b09d9b43576cb",
    },
    "flight_delays": {
        "urls": [
            "https://github.com/skrub-data/skrub-data-files/raw/refs/heads/main/flight_delays.zip",
            "https://osf.io/download/45xu3",
        ],
        "sha256": "f26ed72db5792dba3c6f0c32bdd83438e49b1e6e007a6e4e467f805207b2e4ab",
    },
    "medical_charge": {
        "urls": [
            "https://github.com/skrub-data/skrub-data-files/raw/refs/heads/main/medical_charge.zip",
            "https://osf.io/download/g8cvw",
        ],
        "sha256": "4850651103b7c7580587aafaccc05ca7a31125767d4da662e87890346f984b93",
    },
    "midwest_survey": {
        "urls": [
            "https://github.com/skrub-data/skrub-data-files/raw/refs/heads/main/midwest_survey.zip",
            "https://osf.io/download/aedqu",
        ],
        "sha256": "94d5005402e5e72c2d5ce62f4d3115742dd12190db85920159b2ed8f44df7fc2",
    },
    "movielens": {
        "urls": [
            "https://github.com/skrub-data/skrub-data-files/raw/refs/heads/main/movielens.zip",
            "https://osf.io/download/z5yqv",
        ],
        "sha256": "d6b22c707f9a1605da5616ac1a601f4090467c1a02fa663195a42cf80f32fd57",
    },
    "open_payments": {
        "urls": [
            "https://github.com/skrub-data/skrub-data-files/raw/refs/heads/main/open_payments.zip",
            "https://osf.io/download/a7p9w",
        ],
        "sha256": "ead65dcb8d45ec16ab30dd71025c3cfc5730128f85eeb19ce6f56670923f04ba",
    },
    "road_safety": {
        "urls": [
            "https://github.com/skrub-data/skrub-data-files/raw/refs/heads/main/road_safety.zip",
            "https://osf.io/download/bae6d",
        ],
        "sha256": "035df2a644ba2be52022aa9ca5f41790a24cbd9c76434c3e5224c8c218cf6f87",
    },
    "toxicity": {
        "urls": [
            "https://github.com/skrub-data/skrub-data-files/raw/refs/heads/main/toxicity.zip",
            "https://osf.io/download/zebm7",
        ],
        "sha256": "ee187c119925ea4cdb9abd7f0f3758159f042e71b172cafe5b784d79c7590ce3",
    },
    "traffic_violations": {
        "urls": [
            "https://github.com/skrub-data/skrub-data-files/raw/refs/heads/main/traffic_violations.zip",
            "https://osf.io/download/av4cw",
        ],
        "sha256": "b52a145a34b1866b6deee7cbfd1c0b8d2af3bbd53fb5658b155f752ac7d85ce0",
    },
    "videogame_sales": {
        "urls": [
            "https://github.com/skrub-data/skrub-data-files/raw/refs/heads/main/videogame_sales.zip",
            "https://osf.io/download/g2fw4",
        ],
        "sha256": "3e6d995af025b8a3a1dc64983aa9d53c3c6e72150644d42c58c8b86888c3dacd",
    },
}


def get_data_home(data_home=None):
    """Returns the path of the skrub data directory.

    This folder is used by some large dataset loaders to avoid downloading the
    data several times.

    By default the data directory is set to a folder named 'skrub_data' in the
    user home folder.

    You can even customize the default data directory by setting in your environment
    the `SKRUB_DATA_DIRECTORY` variable to an *absolute directory path*.

    Alternatively, it can be set programmatically by giving an explicit folder
    path. The '~' symbol is expanded to the user home folder.

    If the folder does not already exist, it is automatically created.

    Parameters
    ----------
    data_home : pathlib.Path or string, optional
        The path to the skrub data directory. If `None`, the default path
        is `~/skrub_data`.

    Returns
    -------
    data_home : pathlib.Path
        The validated path to the skrub data directory.
    """
    if data_home is not None:
        data_home = Path(data_home)

        # Replace any "~" by the user's home directory
        # https://docs.python.org/3/library/pathlib.html#pathlib.Path.expanduser
        data_home = data_home.expanduser()

        # Resolve relative path to absolute path
        # https://docs.python.org/3/library/pathlib.html#pathlib.Path.resolve
        data_home = data_home.resolve()
    else:
        data_home_envar = os.environ.get(DATA_HOME_ENVAR_NAME)

        if data_home_envar and (path := Path(data_home_envar)).is_absolute():
            data_home = path
        else:
            data_home = Path.home() / "skrub_data"

    data_home.mkdir(parents=True, exist_ok=True)

    return data_home


def get_data_dir(name=None, data_home=None):
    """
    Returns the directory in which skrub looks for data.

    This is typically useful for the end-user to check
    where the data is downloaded and stored.

    Parameters
    ----------
    name : str, optional
        Subdirectory name. If omitted, the root data directory is returned.
    data_home : pathlib.Path or str, optional
        The path to skrub data directory. If `None`, the default path
        is `~/skrub_data`.
    """
    data_dir = get_data_home(data_home)
    if name is not None:
        data_dir = data_dir / name
    return data_dir


def load_simple_dataset(dataset_name, data_home=None):
    """Load a dataframe and its metadata based on its dataset_name.

    The data will be downloaded if not found locally.
    For e.g. the credit_fraud dataset, the filesystem will look like:

    <data_home>/
        fraud/
            fraud.zip
            fraud/
                baskets.csv
                products.csv
                metadata.json

    Parameters
    ----------
    dataset_name : str
        The name of the dataset to load. The name must be a key of `DATASET_INFO`.

    data_home : path, default=None
        The directory where to download and unpack a zip file. If None, 'skrub_data'
        is used.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - <dataset_name> : pd.DataFrame, the dataframe
        - X : pd.DataFrame, features, i.e. the dataframe without the target labels
        - y : pd.DataFrame, target labels
        - metadata : a dictionary containing the name, description, source and target
          (description and source may be missing)
    """
    bunch = load_dataset_files(dataset_name, data_home)
    bunch["X"] = bunch[dataset_name]
    if (target := bunch.metadata.get("target", None)) is not None:
        bunch["y"] = bunch["X"][target]
        bunch["X"] = bunch["X"].drop(columns=target)
    return bunch


def load_dataset_files(dataset_name, data_home):
    data_home = get_data_home(data_home)
    dataset_dir = data_home / dataset_name
    datafiles_dir = dataset_dir / dataset_name

    archive_path = dataset_dir / f"{dataset_name}.zip"
    metadata = DATASET_INFO[dataset_name]

    if archive_path.exists():
        expected_checksum = metadata["sha256"]
        checksum = _sha256(archive_path)
        if expected_checksum != checksum:
            warnings.warn(
                f"SHA256 checksum of existing local file {archive_path.name} "
                f"({checksum}) differs from expected ({expected_checksum}): "
                f"re-downloading from {metadata['urls']} ."
            )
            try:
                shutil.rmtree(datafiles_dir)
            except FileNotFoundError:
                pass
            archive_path.unlink()

    if not datafiles_dir.exists():
        _extract_archive(dataset_dir, archive_path)

    bunch = Bunch()
    for file_path in datafiles_dir.iterdir():
        if file_path.suffix == ".csv":
            bunch[file_path.stem] = pd.read_csv(file_path)
        elif file_path.suffix == ".json":
            bunch[file_path.stem] = json.loads(file_path.read_text(encoding="utf-8"))

    return bunch


def _extract_archive(dataset_dir, archive_path):
    dataset_name = dataset_dir.name

    if not archive_path.exists():
        _download_archive(dataset_name, archive_path)

    try:
        temp_dir = tempfile.mkdtemp(dir=dataset_dir)
        shutil.unpack_archive(archive_path, temp_dir, format="zip")
        path_source = Path(temp_dir) / dataset_name
        path_source.rename(dataset_dir / dataset_name)
    except (Exception, KeyboardInterrupt):
        try:
            archive_path.unlink()
        except Exception:
            pass
        raise
    finally:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


def _download_archive(
    dataset_name,
    archive_path,
    retry=3,
    delay=1,
    timeout=3,
    chunk_size=4096,
):
    metadata = DATASET_INFO[dataset_name]
    remote_checksum = metadata["sha256"]

    for idx in range(1, retry + 1):
        for target_url in metadata["urls"]:
            print(
                f"Downloading {dataset_name!r} from {target_url} (attempt"
                f" {idx}/{retry})"
            )
            try:
                _stream_download(
                    archive_path,
                    target_url,
                    timeout,
                    chunk_size,
                    remote_checksum,
                )
                return
            except Exception as e:
                print(repr(e))

        time.sleep(delay)
        delay *= 3

    else:
        raise OSError(
            f"Can't download the file {dataset_name!r} from urls {metadata['urls']}."
        )


def _stream_download(
    archive_path,
    target_url,
    timeout,
    chunk_size,
    remote_checksum,
):
    dataset_dir = archive_path.parent
    dataset_dir.mkdir(exist_ok=True)

    # We don't use `NamedTemporaryFile` because if the download is successful we
    # want to rename the temp file rather than removing it
    temp_file, temp_file_path = tempfile.mkstemp(
        prefix=archive_path.stem + ".part_", dir=dataset_dir
    )
    os.close(temp_file)

    try:
        temp_file_path = Path(temp_file_path)
        with open(temp_file_path, "wb") as tf:
            response = requests.get(target_url, timeout=timeout, stream=True)
            for chunk in response.iter_content(chunk_size):
                tf.write(chunk)

        if _sha256(temp_file_path) != remote_checksum:
            raise OSError(
                f"File {archive_path.stem!r} checksum verification has failed."
            )
        temp_file_path.rename(archive_path)
    finally:
        try:
            temp_file_path.unlink()
        except Exception:
            pass


def _sha256(path):
    """Calculate the sha256 hash of the file at path."""
    sha256hash = hashlib.sha256()
    chunk_size = 8192
    with open(path, "rb") as f:
        while buffer := f.read(chunk_size):
            sha256hash.update(buffer)
    return sha256hash.hexdigest()
