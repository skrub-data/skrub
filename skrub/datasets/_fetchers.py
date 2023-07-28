"""
Implements the public fetchers from the private API available in `_fetching.py`
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from skrub.datasets._fetching import (
    _fetch_figshare,
    _fetch_openml_dataset,
    _fetch_world_bank_data,
    _resolve_path,
)


@dataclass(unsafe_hash=True)
class Dataset:
    """Represents a dataset and its information.

    Parameters
    ----------

    name : str
        The name of the dataset.
    description : str
        The description of the dataset.
    source : str
        The URL where the dataset originates from.
    X : pd.DataFrame
        The features.
    y : pd.Series or None
        The target, if applicable.
    target : str or None
        The label of the target (`y`), if applicable.
    problem: {"classification", "regression"} or None
        The type of problem the dataset is meant to solve.
        None if unknown.
    """

    name: str
    description: str
    source: str
    X: pd.DataFrame
    y: pd.Series | None
    target: str | None
    problem: Literal["classification", "regression"] | None

    def __eq__(self, other: Dataset) -> bool:
        """
        Implemented for the tests to work without bloating the code.
        The main reason for which it's needed is that equality between
        DataFrame (`X` and `y`) is often ambiguous and will raise an error.
        """
        return (
            self.name == other.name
            and self.description == other.description
            and self.source == other.source
            and self.X.equals(other.X)
            and self.y.equals(other.y)
            and self.target == other.target
            and self.problem == other.problem
        )

    def __repr__(self):
        y = (
            f"{self.y.__class__.__name__} of shape {self.y.shape}"
            if self.y is not None
            else None
        )
        return (
            "Dataset("
            f"name={self.name!r}, "
            f"description={self.description!r}, "
            f"source={self.source!r}, "
            f"X={self.X.__class__.__name__} of shape {self.X.shape}, "
            f"y={y}, "
            f"target={self.target!r}, "
            f"problem={self.problem!r}"
            ")"
        )


ROAD_SAFETY_ID = 42803
OPEN_PAYMENTS_ID = 42738
MIDWEST_SURVEY_ID = 42805
MEDICAL_CHARGE_ID = 42720
EMPLOYEE_SALARIES_ID = 42125
TRAFFIC_VIOLATIONS_ID = 42132
DRUG_DIRECTORY_ID = 43044


def fetch_employee_salaries(
    *,
    drop_linked: bool = True,
    drop_irrelevant: bool = True,
    directory: Path | str | None = None,
) -> Dataset:
    """Fetches the employee salaries dataset, available at https://openml.org/d/42125

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
    directory : pathlib.Path or str, optional
        Directory where the data will be downloaded. If None, the default
        directory, located in the user home folder, is used.

    Returns
    -------
    Dataset
        Dataset object, containing the data and some metadata.
    """
    dataset = Dataset(
        name="Employee salaries",
        problem="regression",
        **_fetch_openml_dataset(
            dataset_id=EMPLOYEE_SALARIES_ID,
            target="current_annual_salary",
            data_directory=_resolve_path(directory, suffix="openml"),
        ),
    )

    if drop_linked:
        dataset.X.drop(
            ["2016_gross_pay_received", "2016_overtime_pay"], axis=1, inplace=True
        )
    if drop_irrelevant:
        dataset.X.drop(["full_name"], axis=1, inplace=True)

    return dataset


def fetch_road_safety(
    *,
    directory: Path | str | None = None,
) -> Dataset:
    """Fetches the road safety dataset, available at https://openml.org/d/42803

    Description of the dataset:
        Data reported to the police about the circumstances of personal injury
        road accidents in Great Britain from 1979, and the maker and model
        information of vehicles involved in the respective accident.
        This version includes data up to 2015.

    Parameters
    ----------
    directory : pathlib.Path or str, optional
        Directory where the data will be downloaded. If None, the default
        directory, located in the user home folder, is used.

    Returns
    -------
    Dataset
        Dataset object, containing the data and some metadata.
    """
    return Dataset(
        name="Road safety",
        problem="classification",
        **_fetch_openml_dataset(
            dataset_id=ROAD_SAFETY_ID,
            data_directory=_resolve_path(directory, suffix="openml"),
            target="Sex_of_Driver",
        ),
    )


def fetch_medical_charge(
    *,
    directory: Path | str | None = None,
) -> Dataset:
    """Fetches the medical charge dataset, available at https://openml.org/d/42720

    Description of the dataset:
        The Inpatient Utilization and Payment Public Use File (Inpatient PUF)
        provides information on inpatient discharges for Medicare
        fee-for-service beneficiaries. The Inpatient PUF includes information
        on utilization, payment (total payment and Medicare payment), and
        hospital-specific charges for the more than 3,000 U.S. hospitals that
        receive Medicare Inpatient Prospective Payment System (IPPS) payments.
        The PUF is organized by hospital and Medicare Severity Diagnosis
        Related Group (MS-DRG) and covers Fiscal Year (FY) 2011 through FY 2016.

    Parameters
    ----------
    directory : pathlib.Path or str, optional
        Directory where the data will be downloaded. If None, the default
        directory, located in the user home folder, is used.

    Returns
    -------
    Dataset
        Dataset object, containing the data and some metadata.
    """
    return Dataset(
        name="Medical charge",
        problem="regression",
        **_fetch_openml_dataset(
            dataset_id=MEDICAL_CHARGE_ID,
            data_directory=_resolve_path(directory, suffix="openml"),
            target="Average_Total_Payments",
        ),
    )


def fetch_midwest_survey(
    *,
    directory: Path | str | None = None,
) -> Dataset:
    """Fetches the midwest survey dataset, available at https://openml.org/d/42805

    Description of the dataset:
        Survey to know if people self-identify as Midwesterners.

    Parameters
    ----------
    directory : pathlib.Path or str, optional
        Directory where the data will be downloaded. If None, the default
        directory, located in the user home folder, is used.

    Returns
    -------
    Dataset
        Dataset object, containing the data and some metadata.
    """
    return Dataset(
        name="Midwest survey",
        problem="classification",
        **_fetch_openml_dataset(
            dataset_id=MIDWEST_SURVEY_ID,
            data_directory=_resolve_path(directory, suffix="openml"),
            target="Census_Region",
        ),
    )


def fetch_open_payments(
    *,
    directory: Path | str | None = None,
) -> Dataset:
    """Fetches the open payments dataset, available at https://openml.org/d/42738

    Description of the dataset:
        Payments given by healthcare manufacturing companies to medical doctors
        or hospitals.

    Parameters
    ----------
    directory : pathlib.Path or str, optional
        Directory where the data will be downloaded. If None, the default
        directory, located in the user home folder, is used.

    Returns
    -------
    Dataset
        Dataset object, containing the data and some metadata.
    """
    return Dataset(
        name="Open payments",
        problem="classification",
        **_fetch_openml_dataset(
            dataset_id=OPEN_PAYMENTS_ID,
            data_directory=_resolve_path(directory, suffix="openml"),
            target="status",
        ),
    )


def fetch_traffic_violations(
    *,
    directory: Path | str | None = None,
) -> Dataset:
    """Fetches the traffic violations dataset, available at https://openml.org/d/42132

    Description of the dataset:
        This dataset contains traffic violation information from all electronic
        traffic violations issued in the Montgomery County, MD. Any information
        that can be used to uniquely identify the vehicle, the vehicle owner or
        the officer issuing the violation will not be published.

    Parameters
    ----------
    directory : pathlib.Path or str, optional
        Directory where the data will be downloaded. If None, the default
        directory, located in the user home folder, is used.

    Returns
    -------
    Dataset
        Dataset object, containing the data and some metadata.
    """
    return Dataset(
        name="Traffic violations",
        problem="classification",
        **_fetch_openml_dataset(
            dataset_id=TRAFFIC_VIOLATIONS_ID,
            data_directory=_resolve_path(directory, suffix="openml"),
            target="violation_type",
        ),
    )


def fetch_drug_directory(
    *,
    directory: Path | str | None = None,
) -> Dataset:
    """Fetches the drug directory dataset, available at https://openml.org/d/43044

    Description of the dataset:
        Product listing data submitted to the U.S. FDA for all unfinished,
        unapproved drugs.

    Parameters
    ----------
    directory : pathlib.Path or str, optional
        Directory where the data will be downloaded. If None, the default
        directory, located in the user home folder, is used.

    Returns
    -------
    Dataset
        Dataset object, containing the data and some metadata.
    """
    return Dataset(
        name="Drug directory",
        problem="classification",
        **_fetch_openml_dataset(
            dataset_id=DRUG_DIRECTORY_ID,
            data_directory=_resolve_path(directory, suffix="openml"),
            target="PRODUCTTYPENAME",
        ),
    )


def fetch_world_bank_indicator(
    indicator_id: str,
    *,
    directory: Path | str | None = None,
    download_if_missing: bool = True,
) -> Dataset:
    """Fetches a dataset of an indicator from the World Bank open data platform.

    The dataset contains two columns: the indicator value and the country names.

    Parameters
    ----------
    indicator_id : str
        ID of the WorldBank table. A list of all available indicators can be
        found at https://data.worldbank.org/indicator.
    directory : pathlib.Path or str, optional
        Directory where the data will be downloaded. If None, the default
        directory, located in the user home folder, is used.
    download_if_missing : bool, default=True
        Whether to download the data from the Internet if not already on disk.

    Returns
    -------
    Dataset
        Dataset object, containing the data and some metadata.
    """
    return Dataset(
        name=f"World Bank indicator {indicator_id!r}",
        target=None,
        problem=None,
        **_fetch_world_bank_data(
            indicator_id=indicator_id,
            data_directory=_resolve_path(directory, suffix="world_bank"),
            download_if_missing=download_if_missing,
        ),
    )


def fetch_figshare(
    figshare_id: str,
    *,
    directory: Path | str | None = None,
    download_if_missing: bool = True,
) -> Dataset:
    """Fetches a table from figshare.

    Parameters
    ----------
    figshare_id : str
        ID of the table on figshare. The ID can be found in the URL of the
        table, e.g. https://figshare.com/articles/dataset/ID/1234567.
    directory : pathlib.Path or str, optional
        Directory where the data will be downloaded. If None, the default
        directory, located in the user home folder, is used.
    download_if_missing : bool, default=True
        Whether to download the data from the Internet if not already on disk.

    Returns
    -------
    Dataset
        Dataset object, containing the data and some metadata.
    """
    return Dataset(
        name=f"figshare_{figshare_id}",
        target=None,
        problem=None,
        **_fetch_figshare(
            figshare_id=figshare_id,
            data_directory=_resolve_path(directory, suffix="figshare"),
            download_if_missing=download_if_missing,
        ),
    )
