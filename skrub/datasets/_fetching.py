"""
Fetching functions to retrieve example datasets, using fetch_openml.

Public API functions should return either a DatasetInfoOnly or a DatasetAll.
"""

from ._utils import _load_dataset_files, load_simple_dataset

# Ignore lines too long, first docstring lines can't be cut
# flake8: noqa: E501


def fetch_employee_salaries(data_home=None):
    """Fetches the employee salaries dataset (regression), available at \
        https://openml.org/d/42125

    Description of the dataset:
        Annual salary information including gross pay and overtime pay for all
        active, permanent employees of Montgomery County, MD paid in calendar
        year 2016. This information will be published annually each year.

    Parameters
    ----------
    TODO

    Returns
    -------
    TODO
    """
    return load_simple_dataset("employee_salaries", data_home)


def fetch_medical_charge(data_home=None):
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
    return load_simple_dataset("medical_charge", data_home)


def fetch_midwest_survey(data_home=None):
    """Fetches the midwest survey dataset (classification), available at https://openml.org/d/42805

    Description of the dataset:
        Survey to know if people self-identify as Midwesterners.

    Returns
    -------
    TODO
    """
    return load_simple_dataset("midwest_survey", data_home)


def fetch_open_payments(data_home=None):
    """Fetches the open payments dataset (classification), available at https://openml.org/d/42738

    Description of the dataset:
        Payments given by healthcare manufacturing companies to medical doctors
        or hospitals.

    Returns
    -------
    TODO
    """
    return load_simple_dataset("open_payments", data_home)


def fetch_traffic_violations(data_home=None):
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
    return load_simple_dataset("traffic_violations", data_home)


def fetch_drug_directory(data_home=None):
    """Fetches the drug directory dataset (classification), available at https://openml.org/d/43044

    Description of the dataset:
        Product listing data submitted to the U.S. FDA for all unfinished,
        unapproved drugs.

    Returns
    -------
    TODO
    """
    return load_simple_dataset("drug_directory", data_home)


def fetch_credit_fraud(data_home=None):
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
    return _load_dataset_files("credit_fraud", data_home)


def fetch_toxicity(data_home=None):
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
    return load_simple_dataset("toxicity", data_home)


def fetch_movie_lens(data_home=None):
    return _load_dataset_files("movielens", data_home)


def fetch_flight_delays(data_home=None):
    return _load_dataset_files("flight_delays", data_home)


def fetch_country_happiness(data_home=None):
    return _load_dataset_files("country_happiness", data_home)


def fetch_bike_sharing(data_home=None):
    pass
