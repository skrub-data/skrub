"""
Fetching functions to retrieve example datasets, using
Scikit-Learn's ``fetch_openml()`` function.

Public API functions should return either a `DatasetInfoOnly` or a `DatasetAll`.
"""

# Future notes:
# - Watch out for ``fetch_openml()`` API modifications:
# as of january 2021, the function is marked as experimental.

from __future__ import annotations

import gzip
import json
import urllib.request
import warnings
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union
from urllib.error import URLError
from zipfile import BadZipFile, ZipFile

import pandas as pd
from pyarrow.parquet import ParquetFile
from sklearn import __version__ as sklearn_version
from sklearn.datasets import fetch_openml

from dirty_cat._utils import parse_version
from dirty_cat.datasets._utils import get_data_dir

# Directory where the ``.gz`` files containing the
# details on downloaded datasets are stored.
# Note: the tree structure is created by ``fetch_openml()``.
# As of october 2020, this function is annotated as
# ``Experimental`` so the structure might change in future releases.
# This path will be concatenated to the dirty_cat data directory,
# available via the function ``get_data_dir()``.
DETAILS_DIRECTORY: str = "openml/openml.org/api/v1/json/data/"

# Same as above; for the datasets features location.
FEATURES_DIRECTORY: str = "openml/openml.org/api/v1/json/data/features/"

# Same as above; for the datasets data location.
DATA_DIRECTORY: str = "openml/openml.org/data/v1/download/"

# The IDs of the datasets, from OpenML.
# For each dataset, its URL is constructed as follows:
openml_url: str = "https://www.openml.org/d/{ID}"
ROAD_SAFETY_ID: int = 42803
OPEN_PAYMENTS_ID: int = 42738
MIDWEST_SURVEY_ID: int = 42805
MEDICAL_CHARGE_ID: int = 42720
EMPLOYEE_SALARIES_ID: int = 42125
TRAFFIC_VIOLATIONS_ID: int = 42132
DRUG_DIRECTORY_ID: int = 43044


@dataclass(unsafe_hash=True)
class Details:
    name: str
    file_id: str
    description: str


@dataclass(unsafe_hash=True)
class Features:
    names: List[str]


@dataclass(unsafe_hash=True)
class DatasetAll:
    """
    Represents a dataset and its information.
    With this state, the dataset is loaded in memory as a pandas DataFrame
    (``X`` and ``y``).
    Additional information such as `path` and `read_csv_kwargs` are provided
    in case the dataframe has to be read from disk, as such:

    .. code:: python

        ds = fetch_employee_salaries(load_dataframe=False)
        df = pd.read_csv(ds.path, **ds.read_csv_kwargs)

    """

    name: str
    description: str
    source: str
    target: str
    X: pd.DataFrame
    y: pd.Series
    path: Path
    read_csv_kwargs: Dict[str, Any]

    def __eq__(self, other: DatasetAll) -> bool:
        """
        Implemented for the tests to work without bloating the code.
        The main reason for which it's needed is that equality between
        DataFrames (X and y) is often ambiguous and will raise an error.
        """
        return (
            self.name == other.name
            and self.description == other.description
            and self.source == other.source
            and self.target == other.target
            and self.X.equals(other.X)
            and self.y.equals(other.y)
            and self.path == other.path
            and self.read_csv_kwargs == other.read_csv_kwargs
        )


@dataclass(unsafe_hash=True)
class DatasetInfoOnly:
    """
    Represents a dataset and its information.
    With this state, the dataset is NOT loaded in memory, but can be read
    with ``path`` and ``read_csv_kwargs``, as such:

    .. code:: python

        ds = fetch_employee_salaries(load_dataframe=False)
        df = pd.read_csv(ds.path, **ds.read_csv_kwargs)

    """

    name: str
    description: str
    source: str
    target: str
    path: Path
    read_csv_kwargs: Dict[str, Any]


def _fetch_openml_dataset(
    dataset_id: int,
    data_directory: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Gets a dataset from OpenML (https://www.openml.org),
    or from the disk if already downloaded.

    Parameters
    ----------
    dataset_id: int
        The ID of the dataset to fetch.
    data_directory: Path, optional
        A directory to save the data to.
        By default, the dirty_cat data directory.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
          - ``description``: str
              The description of the dataset,
              as gathered from OpenML.
          - ``source``: str
              The dataset's URL from OpenML.
          - ``path``: pathlib.Path
              The local path leading to the dataset,
              saved as a CSV file.

    """
    if data_directory is None:
        data_directory = get_data_dir()

    # Make path absolute
    data_directory = data_directory.resolve()
    data_directory.mkdir(parents=True, exist_ok=True)

    # Construct the path to the gzip file containing the details on a dataset.
    details_gz_path = data_directory / DETAILS_DIRECTORY / f"{dataset_id}.gz"
    features_gz_path = data_directory / FEATURES_DIRECTORY / f"{dataset_id}.gz"

    if not details_gz_path.is_file() or not features_gz_path.is_file():
        # If the details file or the features file don't exist,
        # download the dataset.
        warnings.warn(
            f"Could not find the dataset {dataset_id!r} locally. "
            "Downloading it from OpenML; this might take a while... "
            "If it is interrupted, some files might be invalid/incomplete: "
            "if on the following run, the fetching raises errors, you can try "
            f"fixing this issue by deleting the directory {data_directory!s}.",
            UserWarning,
            stacklevel=2,
        )
        _download_and_write_openml_dataset(
            dataset_id=dataset_id, data_directory=data_directory
        )
    details = _get_details(details_gz_path)

    # The file ID is required because the data file is named after this ID,
    # and not after the dataset's.
    file_id = details.file_id
    csv_path = data_directory / f"{details.name}.csv"

    data_gz_path = data_directory / DATA_DIRECTORY / f"{file_id}.gz"

    if not data_gz_path.is_file():
        # double-check.
        # If the data file does not exist, download the dataset.
        _download_and_write_openml_dataset(
            dataset_id=dataset_id, data_directory=data_directory
        )

    if not csv_path.is_file():
        # If the CSV file does not exist, use the dataset
        # downloaded by ``fetch_openml()`` to construct it.
        features = _get_features(features_gz_path)
        _export_gz_data_to_csv(data_gz_path, csv_path, features)

    url = openml_url.format(ID=dataset_id)

    return {
        "description": details.description,
        "source": url,
        "path": csv_path.resolve(),
    }


def _fetch_world_bank_data(
    indicator_id: str,
    data_directory: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Gets a dataset from World Bank open data platform
    (https://data.worldbank.org/).

    Parameters
    ----------
    indicator_id: str
        The ID of the indicator's dataset to fetch.
    data_directory: Path, optional
        A directory to save the data to.
        By default, the dirty_cat data directory.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
          - ``description``: str
              The description of the dataset,
              as gathered from World Bank data.
          - ``source``: str
              The dataset's URL from the World Bank data platform.
          - ``path``: pathlib.Path
              The local path leading to the dataset,
              saved as a CSV file.

    """
    if data_directory is None:
        data_directory = get_data_dir()

    csv_path = (data_directory / f"{indicator_id}.csv").resolve()
    data_directory.mkdir(parents=True, exist_ok=True)
    url = f"https://api.worldbank.org/v2/en/indicator/{indicator_id}?downloadformat=csv"  # noqa
    if csv_path.is_file():
        df = pd.read_csv(csv_path, nrows=0)
        indicator_name = df.columns[1]
    else:
        warnings.warn(
            f"Could not find the dataset {indicator_id!r} locally. "
            "Downloading it from the World Bank; this might take a while... "
            "If it is interrupted, some files might be invalid/incomplete: "
            "if on the following run, the fetching raises errors, you can try "
            f"fixing this issue by deleting the directory {csv_path!s}.",
            UserWarning,
            stacklevel=2,
        )
        try:
            filehandle, _ = urllib.request.urlretrieve(url)
            zip_file_object = ZipFile(filehandle, "r")
            for name in zip_file_object.namelist():
                if "Metadata" not in name:
                    true_file = name
                    break
            else:
                raise FileNotFoundError(
                    "Could not find any non-metadata file "
                    f"for indicator {indicator_id!r}."
                )
            file = zip_file_object.open(true_file)
        except BadZipFile as e:
            raise FileNotFoundError(
                "Couldn't find csv file, the indicator id "
                f"{indicator_id!r} seems invalid."
            ) from e
        except URLError:
            raise URLError("No internet connection or the website is down.")
        # Read and modify the csv file
        df = pd.read_csv(file, skiprows=3)  # FIXME: why three rows?
        indicator_name = df.iloc[0, 2]
        df[indicator_name] = df.stack().groupby(level=0).last()
        df = df[df[indicator_name] != indicator_id]
        df = df[["Country Name", indicator_name]]

        df.to_csv(csv_path, index=False)
    description = f"This table shows the {indicator_name!r} World Bank indicator."
    return {
        "dataset_name": indicator_name,
        "description": description,
        "source": url,
        "path": csv_path,
    }


def _fetch_figshare(
    figshare_id: str,
    data_directory: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Fetch a dataset from figshare using the download ID number.

    Parameters
    ----------
    figshare_id: str
        The ID of the dataset to fetch.
    data_directory: Path, optional
        A directory to save the data to.
        By default, the dirty_cat data directory.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
          - ``description``: str
              The description of the dataset.
          - ``source``: str
              The dataset's URL.
          - ``path``: pathlib.Path
              The local path leading to the dataset,
              saved as a parquet file.

    Notes
    -----
    The files are read and returned in parquet format, this function needs
    pyarrow installed to run correctly.

    """
    if data_directory is None:
        data_directory = get_data_dir()
    parquet_path = (data_directory / f"figshare_{figshare_id}.parquet").resolve()
    data_directory.mkdir(parents=True, exist_ok=True)
    url = f"https://figshare.com/ndownloader/files/{figshare_id}"
    description = f"This table shows the {figshare_id!r} figshare file."
    file_paths = [
        file
        for file in data_directory.iterdir()
        if file.name.startswith(f"figshare_{figshare_id}")
    ]
    if len(file_paths) > 0:
        if len(file_paths) == 1:
            parquet_paths = [str(file_paths[0].resolve())]
        else:
            parquet_paths = []
            for path in file_paths:
                parquet_path = str(path.resolve())
                parquet_paths += [parquet_path]
        return {
            "dataset_name": figshare_id,
            "description": description,
            "source": url,
            "path": parquet_paths,
        }
    else:
        warnings.warn(
            f"Could not find the dataset {figshare_id!r} locally. "
            "Downloading it from figshare; this might take a while... "
            "If it is interrupted, some files might be invalid/incomplete: "
            "if on the following run, the fetching raises errors, you can try "
            f"fixing this issue by deleting the directory {parquet_path!s}.",
            UserWarning,
            stacklevel=2,
        )
        try:
            filehandle, _ = urllib.request.urlretrieve(url)
            df = ParquetFile(filehandle)
            record = df.iter_batches(
                batch_size=1_000_000,
            )
            idx = []
            for _, x in enumerate(
                chain(range(0, df.metadata.num_rows, 1_000_000), [df.metadata.num_rows])
            ):
                idx += [x]
            parquet_paths = []
            for i in range(1, len(idx)):
                parquet_path = (
                    data_directory / f"figshare_{figshare_id}_{idx[i]}.parquet"
                ).resolve()
                batch = next(record).to_pandas()
                batch.to_parquet(parquet_path, index=False)
                parquet_paths += [parquet_path]
            return {
                "dataset_name": figshare_id,
                "description": description,
                "source": url,
                "path": parquet_paths,
            }
        except URLError:
            raise URLError("No internet connection or the website is down.")


def _download_and_write_openml_dataset(dataset_id: int, data_directory: Path) -> None:
    """
    Downloads a dataset from OpenML,
    taking care of creating the directories.

    Parameters
    ----------
    dataset_id: int
        The ID of the dataset to download.
    data_directory: Path
        The directory in which the data will be saved.

    Raises
    ------
    ValueError
        If the ID is incorrect (does not exist on OpenML)
    urllib.error.URLError
        If there is no Internet connection.

    """
    # The ``fetch_openml()`` function returns a Scikit-Learn ``Bunch`` object,
    # which behaves just like a ``namedtuple``.
    # However, we do not want to save this data into memory:
    # we will read it from the disk later.
    kwargs = {}
    if parse_version("1.2") <= parse_version(sklearn_version) < parse_version("1.2.2"):
        # Avoid the warning, but don't use auto yet because of
        # https://github.com/scikit-learn/scikit-learn/issues/25478
        kwargs.update(
            {
                "parser": "liac-arff",
            }
        )
    elif parse_version(sklearn_version) >= parse_version("1.2.2"):
        kwargs.update(
            {
                "parser": "auto",
            }
        )
    fetch_openml(
        data_id=dataset_id,
        data_home=str(data_directory),
        as_frame=True,
        **kwargs,
    )


def _read_json_from_gz(compressed_dir_path: Path) -> dict:
    """
    Opens a gzip file, reads its content (JSON expected),
    and returns a dictionary.

    Parameters
    ----------
    compressed_dir_path: Path
        Path to the ``.gz`` file to read.

    Returns
    -------
    dict
        The information contained in the file,
        parsed from plain-text JSON.

    """
    if not compressed_dir_path.is_file():
        raise FileNotFoundError(f"Couldn't find file {compressed_dir_path!s}")

    # Read content
    with gzip.open(compressed_dir_path, mode="rt") as gz:
        content = gz.read()

    details_json = json.JSONDecoder().decode(content)
    return details_json


def _get_details(compressed_dir_path: Path) -> Details:
    """
    Gets useful details from the details file.

    Parameters
    ----------
    compressed_dir_path: Path
        The path to the ``.gz`` file containing the details.

    Returns
    -------
    Details
        A ``Details`` object.

    """
    details = _read_json_from_gz(compressed_dir_path)["data_set_description"]
    # We filter out the irrelevant information.
    # If you want to modify this list (to add or remove items)
    # you must also modify the ``Details`` object definition.
    return Details(
        name=details["name"],
        file_id=details["file_id"],
        description=details["description"],
    )


def _get_features(compressed_dir_path: Path) -> Features:
    """
    Gets features that can be inserted in the CSV file.
    The most important feature being the column names.

    Parameters
    ----------
    compressed_dir_path: Path
        Path to the gzip file containing the features.

    Returns
    -------
    Features
        A ``Features`` object.

    """
    raw_features = _read_json_from_gz(compressed_dir_path)["data_features"]
    # We filter out the irrelevant information.
    # If you want to modify this list (to add or remove items)
    # you must also modify the ``Features`` object definition.
    return Features(names=[column["name"] for column in raw_features["feature"]])


def _export_gz_data_to_csv(
    compressed_dir_path: Path, destination_file: Path, features: Features
) -> None:
    """
    Reads a gzip file containing ARFF data,
    and writes it to a target CSV.

    Parameters
    ----------
    compressed_dir_path: Path
        Path to the ``.gz`` file containing the ARFF data.
    destination_file: Path
        A CSV file to write to.
    features: Features
        A ``Features`` object containing the first CSV line (the column names).

    """
    atdata_found = False
    with destination_file.open(mode="w", encoding="utf8") as csv:
        with gzip.open(compressed_dir_path, mode="rt", encoding="utf8") as gz:
            gz: TextIO  # Clarify for IDEs
            csv.write(_features_to_csv_format(features))
            csv.write("\n")
            # We will look at each line of the file until we find
            # "@data": only after this tag is the actual CSV data.
            for line in gz.readlines():
                if not atdata_found:
                    if line.lower().startswith("@data"):
                        atdata_found = True
                else:
                    csv.write(line)


def _features_to_csv_format(features: Features) -> str:
    return ",".join(features.names)


def _fetch_dataset_as_dataclass(
    source: str,
    dataset_name: str,
    dataset_id: Union[int, str],
    target: Optional[str],
    load_dataframe: bool,
    data_directory: Optional[Union[Path, str]] = None,
    read_csv_kwargs: Optional[dict] = None,
) -> Union[DatasetAll, DatasetInfoOnly]:
    """
    Takes a dataset identifier, a target column name (if applicable),
    and some additional keyword arguments for `pd.read_csv`.

    If you don't need the dataset to be loaded in memory,
    pass `load_dataframe=False`.

    To save/load the dataset to/from a specific directory,
    pass `data_directory`. If left default (None), uses the default dirty_cat
    data directory.

    For loading data from:
    - the World Bank platform, specify `source="world_bank"`.
    - OpenML, specify `source="openml"`

    If the dataset doesn't have a target (unsupervised learning or inapplicable),
    explicitly specify `target=None`.

    Returns
    -------
    DatasetAll
        If `load_dataframe=True`

    DatasetInfoOnly
        If `load_dataframe=False`

    """
    if isinstance(data_directory, str):
        data_directory = Path(data_directory)

    if source == "openml":
        info = _fetch_openml_dataset(dataset_id, data_directory)
    elif source == "world_bank":
        info = _fetch_world_bank_data(dataset_id, data_directory)
    elif source == "figshare":
        info = _fetch_figshare(dataset_id, data_directory)
    else:
        raise ValueError(f"Unknown source {source!r}")

    if read_csv_kwargs is None:
        read_csv_kwargs = {}

    if target is None:
        target = []

    if load_dataframe:
        if source == "figshare":
            df = pd.read_parquet(info["path"])
        else:
            df = pd.read_csv(info["path"], **read_csv_kwargs)
        y = df[target]
        X = df.drop(target, axis="columns")
        dataset = DatasetAll(
            name=dataset_name,
            description=info["description"],
            source=info["source"],
            target=target,
            X=X,
            y=y,
            path=info["path"],
            read_csv_kwargs=read_csv_kwargs,
        )
    else:
        dataset = DatasetInfoOnly(
            name=dataset_name,
            description=info["description"],
            source=info["source"],
            target=target,
            path=info["path"],
            read_csv_kwargs=read_csv_kwargs,
        )

    return dataset


# Datasets fetchers section
# Public API


def fetch_employee_salaries(
    load_dataframe: bool = True,
    drop_linked: bool = True,
    drop_irrelevant: bool = True,
    directory: Optional[Union[Path, str]] = None,
) -> Union[DatasetAll, DatasetInfoOnly]:
    """Fetches the employee_salaries dataset (regression), available at
    https://openml.org/d/42125

    Description of the dataset:
        Annual salary information including gross pay and overtime pay for all
        active, permanent employees of Montgomery County, MD paid in calendar
        year 2016. This information will be published annually each year.

    Parameters
    ----------
    drop_linked: bool (default True)
        Drops columns "2016_gross_pay_received" and "2016_overtime_pay",
        which are closely linked to "current_annual_salary", the target.

    drop_irrelevant: bool (default True)
        Drops column "full_name", which is usually irrelevant to the
        statistical analysis.

    Returns
    -------
    DatasetAll
        If `load_dataframe=True`

    DatasetInfoOnly
        If `load_dataframe=False`
    """
    dataset = _fetch_dataset_as_dataclass(
        source="openml",
        dataset_name="Employee salaries",
        dataset_id=EMPLOYEE_SALARIES_ID,
        target="current_annual_salary",
        read_csv_kwargs={
            "quotechar": "'",
            "escapechar": "\\",
            "na_values": ["?"],
        },
        load_dataframe=load_dataframe,
        data_directory=directory,
    )
    if load_dataframe:
        if drop_linked:
            dataset.X.drop(
                ["2016_gross_pay_received", "2016_overtime_pay"], axis=1, inplace=True
            )
        if drop_irrelevant:
            dataset.X.drop(["full_name"], axis=1, inplace=True)

    return dataset


def fetch_road_safety(
    load_dataframe: bool = True,
    directory: Optional[Union[Path, str]] = None,
) -> Union[DatasetAll, DatasetInfoOnly]:
    """Fetches the road safety dataset (classification), available at
    https://openml.org/d/42803

    Description of the dataset:
        Data reported to the police about the circumstances of personal injury
        road accidents in Great Britain from 1979, and the maker and model
        information of vehicles involved in the respective accident.
        This version includes data up to 2015.

    Returns
    -------
    DatasetAll
        If `load_dataframe=True`

    DatasetInfoOnly
        If `load_dataframe=False`
    """
    return _fetch_dataset_as_dataclass(
        source="openml",
        dataset_name="Road safety",
        dataset_id=ROAD_SAFETY_ID,
        target="Sex_of_Driver",
        read_csv_kwargs={
            "na_values": ["?"],
        },
        load_dataframe=load_dataframe,
        data_directory=directory,
    )


def fetch_medical_charge(
    load_dataframe: bool = True,
    directory: Optional[Union[Path, str]] = None,
) -> Union[DatasetAll, DatasetInfoOnly]:
    """Fetches the medical charge dataset (regression), available at
    https://openml.org/d/42720

    Description of the dataset:
        The Inpatient Utilization and Payment Public Use File (Inpatient PUF)
        provides information on inpatient discharges for Medicare
        fee-for-service beneficiaries. The Inpatient PUF includes information
        on utilization, payment (total payment and Medicare payment), and
        hospital-specific charges for the more than 3,000 U.S. hospitals that
        receive Medicare Inpatient Prospective Payment System (IPPS) payments.
        The PUF is organized by hospital and Medicare Severity Diagnosis
        Related Group (MS-DRG) and covers Fiscal Year (FY) 2011 through FY 2016.

    Returns
    -------
    DatasetAll
        If `load_dataframe=True`

    DatasetInfoOnly
        If `load_dataframe=False`
    """
    return _fetch_dataset_as_dataclass(
        source="openml",
        dataset_name="Medical charge",
        dataset_id=MEDICAL_CHARGE_ID,
        target="Average_Total_Payments",
        read_csv_kwargs={
            "quotechar": "'",
            "escapechar": "\\",
        },
        load_dataframe=load_dataframe,
        data_directory=directory,
    )


def fetch_midwest_survey(
    load_dataframe: bool = True,
    directory: Optional[Union[Path, str]] = None,
) -> Union[DatasetAll, DatasetInfoOnly]:
    """Fetches the midwest survey dataset (classification), available at
    https://openml.org/d/42805

    Description of the dataset:
        Survey to know if people self-identify as Midwesterners.

    Returns
    -------
    DatasetAll
        If `load_dataframe=True`

    DatasetInfoOnly
        If `load_dataframe=False`
    """
    return _fetch_dataset_as_dataclass(
        source="openml",
        dataset_name="Midwest survey",
        dataset_id=MIDWEST_SURVEY_ID,
        target="Census_Region",
        read_csv_kwargs={
            "quotechar": "'",
            "escapechar": "\\",
        },
        load_dataframe=load_dataframe,
        data_directory=directory,
    )


def fetch_open_payments(
    load_dataframe: bool = True,
    directory: Optional[Union[Path, str]] = None,
) -> Union[DatasetAll, DatasetInfoOnly]:
    """Fetches the open payments dataset (classification), available at
    https://openml.org/d/42738

    Description of the dataset:
        Payments given by healthcare manufacturing companies to medical doctors
        or hospitals.

    Returns
    -------
    DatasetAll
        If `load_dataframe=True`

    DatasetInfoOnly
        If `load_dataframe=False`
    """
    return _fetch_dataset_as_dataclass(
        source="openml",
        dataset_name="Open payments",
        dataset_id=OPEN_PAYMENTS_ID,
        target="status",
        read_csv_kwargs={
            "quotechar": "'",
            "escapechar": "\\",
            "na_values": ["?"],
        },
        load_dataframe=load_dataframe,
        data_directory=directory,
    )


def fetch_traffic_violations(
    load_dataframe: bool = True,
    directory: Optional[Union[Path, str]] = None,
) -> Union[DatasetAll, DatasetInfoOnly]:
    """Fetches the traffic violations dataset (classification), available at
    https://openml.org/d/42132

    Description of the dataset:
        This dataset contains traffic violation information from all electronic
        traffic violations issued in the Montgomery County, MD. Any information
        that can be used to uniquely identify the vehicle, the vehicle owner or
        the officer issuing the violation will not be published.

    Returns
    -------
    DatasetAll
        If `load_dataframe=True`

    DatasetInfoOnly
        If `load_dataframe=False`
    """
    return _fetch_dataset_as_dataclass(
        source="openml",
        dataset_name="Traffic violations",
        dataset_id=TRAFFIC_VIOLATIONS_ID,
        target="violation_type",
        read_csv_kwargs={
            "quotechar": "'",
            "escapechar": "\\",
            "na_values": ["?"],
        },
        load_dataframe=load_dataframe,
        data_directory=directory,
    )


def fetch_drug_directory(
    load_dataframe: bool = True,
    directory: Optional[Union[Path, str]] = None,
) -> Union[DatasetAll, DatasetInfoOnly]:
    """Fetches the drug directory dataset (classification), available at
    https://openml.org/d/43044

    Description of the dataset:
        Product listing data submitted to the U.S. FDA for all unfinished,
        unapproved drugs.

    Returns
    -------
    DatasetAll
        If `load_dataframe=True`

    DatasetInfoOnly
        If `load_dataframe=False`
    """
    return _fetch_dataset_as_dataclass(
        source="openml",
        dataset_name="Drug directory",
        dataset_id=DRUG_DIRECTORY_ID,
        target="PRODUCTTYPENAME",
        read_csv_kwargs={
            "quotechar": "'",
            "escapechar": "\\",
        },
        load_dataframe=load_dataframe,
        data_directory=directory,
    )


def fetch_world_bank_indicator(
    indicator_id: str,
    load_dataframe: bool = True,
    directory: Optional[Union[Path, str]] = None,
) -> Union[DatasetAll, DatasetInfoOnly]:
    """Fetches a dataset of an indicator from the World Bank
    open data platform.

     Description of the dataset:
     > The dataset contains two columns: the indicator value and the
       country names. A list of all available indicators can be found
       at https://data.worldbank.org/indicator.

     Returns
     -------
     DatasetAll
         If `load_dataframe=True`

     DatasetInfoOnly
         If `load_dataframe=False`
    """
    return _fetch_dataset_as_dataclass(
        source="world_bank",
        dataset_name=f"World Bank indicator {indicator_id!r}",
        dataset_id=indicator_id,
        target=None,
        load_dataframe=load_dataframe,
        data_directory=directory,
    )


def fetch_figshare(
    figshare_id: str,
    load_dataframe: bool = True,
    directory: Optional[Path] = None,
) -> Union[DatasetAll, DatasetInfoOnly]:
    """Fetches a table of from figshare.

    Returns
    -------
    DatasetAll
        If `load_dataframe=True`

    DatasetInfoOnly
        If `load_dataframe=False`
    """
    return _fetch_dataset_as_dataclass(
        source="figshare",
        dataset_name=f"figshare_{figshare_id!r}",
        dataset_id=figshare_id,
        target=None,
        load_dataframe=load_dataframe,
        data_directory=directory,
    )
