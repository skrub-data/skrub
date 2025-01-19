"""
Fetching functions to retrieve example datasets, using fetch_openml.

Public API functions should return either a DatasetInfoOnly or a DatasetAll.
"""

# Future notes:
# - Watch out for ``fetch_openml()`` API modifications:
# as of january 2021, the function is marked as experimental.

from __future__ import annotations

import gzip
import json
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Literal, TextIO
from urllib.error import URLError
from zipfile import BadZipFile, ZipFile

import pandas as pd
from sklearn.utils import Bunch

from skrub.datasets._base import get_data_dir
from skrub.datasets._figshare import fetch_figshare
from skrub.datasets._openml import fetch_openml_skb

# Ignore lines too long, first docstring lines can't be cut
# flake8: noqa: E501


ROAD_SAFETY_ID: int = 42803
OPEN_PAYMENTS_ID: int = 42738
MIDWEST_SURVEY_ID: int = 42805
MEDICAL_CHARGE_ID: int = 42720
EMPLOYEE_SALARIES_ID: int = 42125
TRAFFIC_VIOLATIONS_ID: int = 42132
DRUG_DIRECTORY_ID: int = 43044

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/{zip_directory}.zip"

# A dictionary storing the sha256 hashes of the figshare files
figshare_id_to_hash = {
    39142985: "47d73381ef72b050002a8642194c6718a4954ec9e6c556f4c4ddc6ed84ceec92",
    39149066: "e479cf9741a90c40401697e7fa54409e3b9cfa09f27502877382e64e86fbfcd0",
    39149069: "7b0dcdb15d3aeecba6022c929665ee064f6fb4b8b94186a6e89b6fbc781b3775",
    39149072: "4f58f15168bb8a6cc8b152bd48995bc7d1a4d4d89a9e22d87aa51ccf30118122",
    39149075: "7037603362af1d4bf73551d50644c0957cb91d2b4892e75413f57f415962029a",
    39254360: "531130c714ba6ee9902108d4010f42388aa9c0b3167d124cd57e2c632df3e05a",
    39266300: "37b23b2c37a1f7ff906bc7951cbed4be15d8417dad0762092282f7b491cf8c21",
    39266678: "4e041322985e078de8b08acfd44b93a5ce347c1e501e9d869651e753de747ba1",
    40019230: "4d43fed75dba1e59a5587bf31c1addf2647a1f15ebea66e93177ccda41e18f2f",
    40019788: "67ae86496c8a08c6cc352f573160a094f605e7e0da022eb91c603abb7edf3747",
}


def _fetch_world_bank_data(
    indicator_id: str,
    data_directory: Path | None = None,
) -> dict[str, Any]:
    """Gets a dataset from World Bank open data platform (https://data.worldbank.org/).

    Parameters
    ----------
    indicator_id : str
        The ID of the indicator's dataset to fetch.
    data_directory : pathlib.Path, optional
        The directory where the dataset is stored.
        By default, a subdirectory "world_bank" in the skrub data directory.

    Returns
    -------
    mapping of str to any
        A dictionary containing:
          - `description` : str
              The description of the dataset,
              as gathered from World Bank data.
          - `source` : str
              The dataset's URL from the World Bank data platform.
          - `path` : pathlib.Path
              The local path leading to the dataset,
              saved as a CSV file.
    """
    if data_directory is None:
        data_directory = get_data_dir(name="world_bank")

    csv_path = (data_directory / f"{indicator_id}.csv").resolve()
    data_directory.mkdir(parents=True, exist_ok=True)
    url = f"https://api.worldbank.org/v2/en/indicator/{indicator_id}?downloadformat=csv"
    if csv_path.is_file():
        df = pd.read_csv(csv_path, nrows=0)
        indicator_name = df.columns[1]
    else:
        warnings.warn(
            (
                f"Could not find the dataset {indicator_id!r} locally. "
                "Downloading it from the World Bank; this might take a while... "
                "If it is interrupted, some files might be invalid/incomplete: "
                "if on the following run, the fetching raises errors, you can try "
                f"fixing this issue by deleting the directory {csv_path!s}."
            ),
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


def _fetch_movielens(dataset_id: str, data_directory: Path | None = None) -> dict[str]:
    """Downloads a subset of the Movielens dataset.

    Parameters
    ----------
    data_directory : :obj:`~pathlib.Path`
        The directory in which the data will be saved.
    """
    if data_directory is None:
        data_directory = get_data_dir()

    options = ["movies", "ratings"]
    if dataset_id not in options:
        raise ValueError(f"dataset_id options are {options}, got '{dataset_id}'.")

    zip_directory = Path("ml-latest-small")
    file_path = data_directory / zip_directory / f"{dataset_id}.csv"
    detail_path = data_directory / zip_directory / "README.txt"
    if not file_path.is_file() or not detail_path.is_file():
        # If the details file or the features file don't exist,
        # download the dataset.
        warnings.warn(
            (
                f"Could not find the dataset {dataset_id!r} locally. "
                "Downloading it from MovieLens; this might take a while... "
                "If it is interrupted, some files might be invalid/incomplete: "
                "if on the following run, the fetching raises errors, you can try "
                f"fixing this issue by deleting the directory {data_directory!s}."
            ),
            UserWarning,
            stacklevel=2,
        )
        _download_and_write_movielens_dataset(
            dataset_id,
            data_directory,
            zip_directory,
        )

    description = open(detail_path).read()

    url = MOVIELENS_URL.format(zip_directory=zip_directory)

    return {
        "description": description,
        "source": url,
        "path": Path(data_directory) / zip_directory / f"{dataset_id}.csv",
    }


def _download_and_write_movielens_dataset(dataset_id, data_directory, zip_directory):
    url = MOVIELENS_URL.format(zip_directory=zip_directory)
    tmp_file = None
    try:
        tmp_file, _ = urllib.request.urlretrieve(url)
        data_file = str((zip_directory / f"{dataset_id}.csv").as_posix())
        readme_file = str((zip_directory / "README.txt").as_posix())
        with ZipFile(tmp_file, "r") as zip_file:
            zip_file.extractall(
                data_directory,
                members=[data_file, readme_file],
            )
    except Exception:
        if tmp_file is not None and Path(tmp_file).exists():
            Path(tmp_file).unlink()
        raise


def _read_json_from_gz(compressed_dir_path: Path) -> dict:
    """Opens a gzip file, reads its content (JSON expected), and returns a dictionary.

    Parameters
    ----------
    compressed_dir_path : pathlib.Path
        Path to the `.gz` file to read.

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
    """Gets useful details from the details file.

    Parameters
    ----------
    compressed_dir_path : pathlib.Path
        The path to the `.gz` file containing the details.

    Returns
    -------
    Details
        A Details instance.
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
    """Gets features that can be inserted in the CSV file.

    The most important feature are the column names.

    Parameters
    ----------
    compressed_dir_path : pathlib.Path
        Path to the gzip file containing the features.

    Returns
    -------
    Features
        A Features instance.
    """
    raw_features = _read_json_from_gz(compressed_dir_path)["data_features"]
    # We filter out the irrelevant information.
    # If you want to modify this list (to add or remove items)
    # you must also modify the ``Features`` object definition.
    return Features(names=[column["name"] for column in raw_features["feature"]])


def _export_gz_data_to_csv(
    compressed_dir_path: Path, destination_file: Path, features: Features
) -> None:
    """Reads a gzip file containing ARFF data, and writes it to a CSV file.

    Parameters
    ----------
    compressed_dir_path : pathlib.Path
        Path to the `.gz` file containing the ARFF data.
    destination_file : pathlib.Path
        A CSV file to write to.
    features : Features
        A Features instance containing the first CSV line (the column names).
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
    source: Literal["openml", "world_bank", "figshare"],
    dataset_name: str,
    dataset_id: int | str,
    target: str | None,
    load_dataframe: bool,
    data_directory: Path | str | None = None,
    read_csv_kwargs: dict | None = None,
) -> DatasetAll | DatasetInfoOnly:
    """Fetches a dataset from a source, and returns it as a dataclass.

    Takes a dataset identifier, a target column name (if applicable),
    and some additional keyword arguments for read_csv.

    If you don't need the dataset to be loaded in memory,
    pass `load_dataframe=False`.

    To save/load the dataset to/from a specific directory,
    pass `data_directory`. If `None`, uses the default skrub
    data directory.

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
        info = (dataset_id, data_directory)
    elif source == "world_bank":
        info = _fetch_world_bank_data(dataset_id, data_directory)
    elif source == "figshare":
        info = _fetch_figshare(dataset_id, data_directory)
    elif source == "movielens":
        info = _fetch_movielens(dataset_id, data_directory)
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


def fetch_employee_salaries(
    *,
    load_dataframe=True,
    drop_linked=True,
    drop_irrelevant=True,
    overload_job_titles=True,
    data_directory=None,
    return_X_y=False,
):
    """Fetches the employee salaries dataset (regression), available at \
        https://openml.org/d/42125

    Description of the dataset:
        Annual salary information including gross pay and overtime pay for all
        active, permanent employees of Montgomery County, MD paid in calendar
        year 2016. This information will be published annually each year.

    Parameters
    ----------
    drop_linked : bool, default=True
        Drops columns "2016_gross_pay_received" and "2016_overtime_pay",
        which are closely linked to "current_annual_salary", the target.

    drop_irrelevant : bool, default=True
        Drops column "full_name", which is usually irrelevant to the
        statistical analysis.

    overload_job_titles : bool, default=True
        Uses the column `underfilled_job_title` to enrich the
        `employee_position_title` column, as it contains more detailed
        information about the job title.

    data_directory: pathlib.Path or str, optional
        The directory where the dataset is stored.

    TODO

    Returns
    -------
    TODO
    """
    data = fetch_openml_skb(
        data_id=EMPLOYEE_SALARIES_ID,
        target_column="current_annual_salary",
        data_home=data_directory,
        return_X_y=return_X_y,
    )
    if return_X_y:
        X = data[0]
    else:
        X = data.data

    if drop_linked:
        X.drop(["2016_gross_pay_received", "2016_overtime_pay"], axis=1, inplace=True)
    if drop_irrelevant:
        X.drop(["full_name"], axis=1, inplace=True)
    if overload_job_titles:
        X["employee_position_title"] = X["underfilled_job_title"].fillna(
            X["employee_position_title"]
        )
        X.drop(labels=["underfilled_job_title"], axis="columns", inplace=True)

    return data


def fetch_road_safety(
    *,
    load_dataframe=True,
    data_directory=None,
    return_X_y=False,
):
    """Fetches the road safety dataset (classification), available at https://openml.org/d/42803

    Description of the dataset:
        Data reported to the police about the circumstances of personal injury
        road accidents in Great Britain from 1979, and the maker and model
        information of vehicles involved in the respective accident.
        This version includes data up to 2015.

    Returns
    -------
    TODO
    """
    return fetch_openml_skb(
        data_id=ROAD_SAFETY_ID,
        data_home=data_directory,
        target_column="Sex_of_Driver",
        return_X_y=return_X_y,
    )


def fetch_medical_charge(
    *,
    load_dataframe,
    data_directory,
    return_X_y=False,
):
    """Fetches the medical charge dataset (regression), available at https://openml.org/d/42720

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
    TODO
    """
    return fetch_openml_skb(
        data_home=data_directory,
        data_id=MEDICAL_CHARGE_ID,
        target_column="Average_Total_Payments",
        return_X_y=return_X_y,
    )


def fetch_midwest_survey(
    *,
    load_dataframe=True,
    data_directory=None,
    return_X_y=False,
):
    """Fetches the midwest survey dataset (classification), available at https://openml.org/d/42805

    Description of the dataset:
        Survey to know if people self-identify as Midwesterners.

    Returns
    -------
    TODO
    """
    return fetch_openml_skb(
        data_id=MIDWEST_SURVEY_ID,
        data_home=data_directory,
        target_column="Census_Region",
        return_X_y=return_X_y,
    )


def fetch_open_payments(
    *,
    load_dataframe=True,
    data_directory=None,
    return_X_y=False,
):
    """Fetches the open payments dataset (classification), available at https://openml.org/d/42738

    Description of the dataset:
        Payments given by healthcare manufacturing companies to medical doctors
        or hospitals.

    Returns
    -------
    TODO
    """
    return fetch_openml_skb(
        data_id=OPEN_PAYMENTS_ID,
        data_home=data_directory,
        target_column="status",
        return_X_y=return_X_y,
    )


def fetch_traffic_violations(
    *,
    load_dataframe=True,
    data_directory=None,
    return_X_y=False,
):
    """Fetches the traffic violations dataset (classification), available at https://openml.org/d/42132

    Description of the dataset:
        This dataset contains traffic violation information from all electronic
        traffic violations issued in the Montgomery County, MD. Any information
        that can be used to uniquely identify the vehicle, the vehicle owner or
        the officer issuing the violation will not be published.

    Returns
    -------
    TODO
    """
    return fetch_openml_skb(
        data_id=TRAFFIC_VIOLATIONS_ID,
        data_home=data_directory,
        target_columns="violation_type",
        return_X_y=return_X_y,
    )


def fetch_drug_directory(
    *,
    load_dataframe=True,
    data_directory=None,
    return_X_y=False,
):
    """Fetches the drug directory dataset (classification), available at https://openml.org/d/43044

    Description of the dataset:
        Product listing data submitted to the U.S. FDA for all unfinished,
        unapproved drugs.

    Returns
    -------
    TODO
    """
    return fetch_openml_skb(
        data_id=DRUG_DIRECTORY_ID,
        data_home=data_directory,
        target_column="PRODUCTTYPENAME",
        return_X_y=return_X_y,
    )


def fetch_world_bank_indicator(
    indicator_id: str,
    *,
    load_dataframe: bool = True,
    data_directory: Path | str | None = None,
) -> DatasetAll | DatasetInfoOnly:
    """Fetches a dataset of an indicator from the World Bank open data platform.

    Description of the dataset:
        The dataset contains two columns: the indicator value and the
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
        data_directory=data_directory,
    )


def fetch_movielens(
    dataset_id: str = "ratings",
    *,
    load_dataframe: bool = True,
    data_directory: Path | None = None,
) -> DatasetAll | DatasetInfoOnly:
    """Fetches a dataset from Movielens.

    Parameters
    ----------
    dataset_id : str
        Either 'ratings' or 'movies'

    Returns
    -------
    :obj:`DatasetAll`
        If `load_dataframe=True`

    :obj:`DatasetInfoOnly`
        If `load_dataframe=False`
    """
    return _fetch_dataset_as_dataclass(
        source="movielens",
        dataset_name="ml-latest-small",
        dataset_id=dataset_id,
        target=None,
        load_dataframe=load_dataframe,
        data_directory=data_directory,
    )


def fetch_credit_fraud(load_dataframe=True, data_directory=None):
    """Fetch the credit fraud dataset from figshare.

    This is an imbalanced binary classification use-case. This dataset consists in
    two tables:

    - baskets, containing the binary fraud target label
    - products

    Baskets contain at least one product each, so aggregation then joining operations
    are required to build a design matrix.

    More details on \
        `Figshare <https://figshare.com/articles/dataset/bnp_fraud_parquet/26892673>`_

    Parameters
    ----------
    load_dataframe : bool, default=True
        Whether or not to load the dataset in memory after download.

    data_directory : str, default=None
        The directory to which the dataset will be written during the download.
        If None, the directory is set to ~/skrub_data.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionnary-like object, whose fields are:
        - product : pd.DataFrame
        - baskets : pd.DataFrame
        - source_product : str
        - source_baskets : str
        - path_product : str
        - path_baskets : str
    """
    dataset_name_to_id = {
        "products": "49176205",
        "baskets": "49176202",
    }
    bunch = Bunch()
    for dataset_name, figshare_id in dataset_name_to_id.items():
        dataset = fetch_figshare(
            figshare_id,
            load_dataframe=load_dataframe,
            data_directory=data_directory,
        )
        bunch[dataset_name] = dataset.X
        bunch[f"source_{dataset_name}"] = dataset.source
        bunch[f"path_{dataset_name}"] = dataset.path

    return bunch


def fetch_toxicity(load_dataframe=True, data_directory=None):
    """Fetch the toxicity dataset from figshare.

    This is a balanced binary classification use-case, where the single table
    consists in only two columns:
    - `text`: the text of the comment
    - `is_toxic`: whether or not the comment is toxic

    Parameters
    ----------
    load_dataframe : bool, default=True
        Whether or not to load the dataset in memory after download.

    data_directory : str, default=None
        The directory to which the dataset will be written during the download.
        If None, the directory is set to ~/skrub_data.

    Returns
    -------
    dataset : DatasetAll
        A dataclass object whose fields are:
        - name: str
        - description: str
        - source: str
        - target: str
        - X: pd.DataFrame
        - y: pd.Series
        - path: Path
    """
    return fetch_figshare(
        figshare_id="49823901",
        load_dataframe=load_dataframe,
        data_directory=data_directory,
        target="is_toxic",
    )
