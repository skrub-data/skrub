"""
Fetching functions to retrieve example datasets from GitHub and OSF.
"""

from ._utils import load_dataset_files, load_simple_dataset


def fetch_employee_salaries(data_home=None):
    """Fetches the employee salaries dataset (regression), available at \
        https://github.com/skrub-data/skrub-data-files

    Description of the dataset:
        Annual salary information including gross pay and overtime pay for all
        active, permanent employees of Montgomery County, MD paid in calendar
        year 2016. This information will be published annually each year.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - employee_salaries : pd.DataFrame, the dataframe
        - X : pd.DataFrame, features, i.e. the dataframe without the target labels
        - y : pd.DataFrame, target labels
        - metadata : a dictionary containing the name, description, source and target
    """
    return load_simple_dataset("employee_salaries", data_home)


def fetch_medical_charge(data_home=None):
    """Fetches the medical charge dataset (regression), available at \
        https://github.com/skrub-data/skrub-data-files

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
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - medical_charge : pd.DataFrame, the dataframe
        - X : pd.DataFrame, features, i.e. the dataframe without the target labels
        - y : pd.DataFrame, target labels
        - metadata : a dictionary containing the name, description, source and target
    """
    return load_simple_dataset("medical_charge", data_home)


def fetch_midwest_survey(data_home=None):
    """Fetches the midwest survey dataset (classification), available at \
        https://github.com/skrub-data/skrub-data-files

    Description of the dataset:
        Survey to know if people self-identify as Midwesterners.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - midwest_survey : pd.DataFrame, the dataframe
        - X : pd.DataFrame, features, i.e. the dataframe without the target labels
        - y : pd.DataFrame, target labels
        - metadata : a dictionary containing the name, description, source and target
    """
    return load_simple_dataset("midwest_survey", data_home)


def fetch_open_payments(data_home=None):
    """Fetches the open payments dataset (classification), available at \
        https://github.com/skrub-data/skrub-data-files

    Description of the dataset:
        Payments given by healthcare manufacturing companies to medical doctors
        or hospitals.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - open_payments : pd.DataFrame, the dataframe
        - X : pd.DataFrame, features, i.e. the dataframe without the target labels
        - y : pd.DataFrame, target labels
        - metadata : a dictionary containing the name, description, source and target
    """
    return load_simple_dataset("open_payments", data_home)


def fetch_traffic_violations(data_home=None):
    """Fetches the traffic violations dataset (classification), available at \
        https://github.com/skrub-data/skrub-data-files

    Description of the dataset:
        This dataset contains traffic violation information from all electronic
        traffic violations issued in the Montgomery County, MD. Any information
        that can be used to uniquely identify the vehicle, the vehicle owner or
        the officer issuing the violation will not be published.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - traffic_violations : pd.DataFrame, the dataframe
        - X : pd.DataFrame, features, i.e. the dataframe without the target labels
        - y : pd.DataFrame, target labels
        - metadata : a dictionary containing the name, description, source and target
    """
    return load_simple_dataset("traffic_violations", data_home)


def fetch_drug_directory(data_home=None):
    """Fetches the drug directory dataset (classification), available at \
        https://github.com/skrub-data/skrub-data-files

    Description of the dataset:
        Product listing data submitted to the U.S. FDA for all unfinished,
        unapproved drugs.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - drug_directory : pd.DataFrame, the dataframe
        - X : pd.DataFrame, features, i.e. the dataframe without the target labels
        - y : pd.DataFrame, target labels
        - metadata : a dictionary containing the name, description, source and target
    """
    return load_simple_dataset("drug_directory", data_home)


def fetch_credit_fraud(data_home=None):
    """Fetch the credit fraud dataset (classification) available at \
        https://github.com/skrub-data/skrub-data-files

    This is an imbalanced binary classification use-case. This dataset consists in
    two tables:

    - baskets, containing the binary fraud target label
    - products

    Baskets contain at least one product each, so aggregation then joining operations
    are required to build a design matrix.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - baskets : pd.DataFrame, table containing baskets ID and target
        - product : pd.DataFrame, table containing features about products contained in
          baskets
        - metadata : a dictionary containing the name, description, source and target
    """
    return load_dataset_files("credit_fraud", data_home)


def fetch_toxicity(data_home=None):
    """Fetch the toxicity dataset (classification) available at \
        https://github.com/skrub-data/skrub-data-files

    This is a balanced binary classification use-case, where the single table
    consists in only two columns:
    - `text`: the text of the comment
    - `is_toxic`: whether or not the comment is toxic

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - toxicity : pd.DataFrame, the dataframe
        - X : pd.DataFrame, features, i.e. the dataframe without the target labels
        - y : pd.DataFrame, target labels
        - metadata : a dictionary containing the name, description, source and target
    """
    return load_simple_dataset("toxicity", data_home)


def fetch_videogame_sales(data_home=None):
    """Fetch the videogame sales dataset (regression) available at \
        https://github.com/skrub-data/skrub-data-files

    This is a regression use-case, where the single table contains information
    about videogames such as the publisher and platform, and the goal is to
    predict the number of sales worldwide.

    .. warning::

        The original dataset is ordered by decreasing number of sales. This
        should be taken into account for cross-validation. Depending on the
        desired setting, one might consider shuffling the rows or ordering by
        publication year and splitting by year.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - videogame_sales : pd.DataFrame, the full dataframe
        - X : pd.DataFrame, features, i.e. the dataframe without the target labels
        - y : pd.DataFrame, target labels
        - metadata : a dictionary containing the name, source and target
    """

    result = load_simple_dataset("videogame_sales", data_home)
    result["X"] = result["X"].drop(
        columns=["Rank", "NA_Sales", "EU_Sales", "JP_Sales", "Other_Sales"]
    )
    return result


def fetch_bike_sharing(data_home=None):
    """Fetch the bike sharing dataset (regression) available at \
        https://github.com/skrub-data/skrub-data-files

    This is a regression use-case, where the goal is to predict demand for a
    bike-sharing service. The features are the dates and holiday and weather
    information.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - bike_sharing : pd.DataFrame, the full dataframe
        - X : pd.DataFrame, features, i.e. the dataframe without the target labels
        - y : pd.DataFrame, target labels
        - metadata : a dictionary containing the name and target
    """

    return load_simple_dataset("bike_sharing", data_home)


def fetch_movielens(data_home=None):
    """Fetch the movielens dataset (regression) available at \
        https://github.com/skrub-data/skrub-data-files

    This is a regression use-case, where the goal is to predict movie ratings.
    More details are provided in the output's ``metadata['description']``.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - movies : pd.DataFrame, movie ID, title and genres
        - ratings: pd.DataFrame, user ID, movie ID, rating
        - metadata : a dictionary containing the name source and description
    """

    return load_dataset_files("movielens", data_home)


def fetch_flight_delays(data_home=None):
    """Fetch the flight delays dataset (regression) available at \
        https://github.com/skrub-data/skrub-data-files

    This is a regression use-case, where the goal is to predict flight delays.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - flights: information about the flights, including departure and
          arrival airports, and delay.
        - airports: information about airports, such as city and coordinates.
          The airport's ``iata`` can be matched to the flights' ``Origin`` and
          ``Dest``.
        - weather: weather data that could be used to help improve the delay
          predictions. Note the weather data is not measured at the airports
          directly but at weather stations, whose location and information is
          provided in ``stations``.
        - stations: information about the weather stations. ``weather`` and
          ``stations`` can be joined on their ``ID`` columns. Weather stations
          can only be matched to the nearest airport based on the latitude and
          longitude.
        - metadata : a dictionary containing the name  of the dataset.
    """
    return load_dataset_files("flight_delays", data_home)


def fetch_country_happiness(data_home=None):
    """Fetch the happiness index dataset (regression) available at \
        https://github.com/skrub-data/skrub-data-files

    This is a regression use-case, where the goal is to predict the happiness
    index. The dataset contains data from the `2022 World Happiness Report
    <https://worldhappiness.report/>`_, and from `the World Bank open data
    platform <https://data.worldbank.org/>`_.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - ``happiness_report``: dataframe, data from the world happiness report
        - ``GDP_per_capita``, ``life_expectancy``, ``legal_rights_index``:
          corresponding tables from the World Bank.
    """
    return load_dataset_files("country_happiness", data_home)
