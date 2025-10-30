"""
Fetching functions to retrieve example datasets from GitHub and OSF.
"""

from ._utils import load_dataset_files, load_simple_dataset


def fetch_employee_salaries(data_home=None, split="all"):
    """Fetches the employee salaries dataset (regression), available at \
        https://github.com/skrub-data/skrub-data-files

    Description of the dataset:
        Annual salary information including gross pay and overtime pay for all
        active, permanent employees of Montgomery County, MD paid in calendar
        year 2016. This dataset is a copy of https://www.openml.org/d/42125
        where some features are dropped to avoid data leaking.
        Size on disk: 1.3MB.

    .. note::

        Some environments like Jupyterlite can run into networking issues when
        connecting to a remote server, but OpenML provides CORS headers. To
        download this dataset using OpenML instead of Github or Figshare, run:

    .. code:: python

        from sklearn.datasets import fetch_openml
        df = fetch_openml(data_id=42125)

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    split : str, default="all"
        The split to load. Can be either "train", "test", or "all".

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - ``employee_salaries`` : pd.DataFrame, the dataframe. Shape: (9228, 9)
        - ``X`` : pd.DataFrame, features, i.e. the dataframe without the target
          labels. Shape: (9228, 8)
        - ``y`` : pd.DataFrame, target labels. Shape: (9228, 1)
        - ``metadata`` : a dictionary containing the name, description, source and
          target
    """
    if split not in ["train", "test", "all"]:
        raise ValueError(
            f"`split` must be one of ['train', 'test', 'all'], got: {split!r}."
        )
    dataset = load_simple_dataset("employee_salaries", data_home)

    id_split = 8000
    if split == "train":
        dataset["employee_salaries"] = dataset["employee_salaries"][:id_split]
        dataset["X"] = dataset["X"][:id_split]
        dataset["y"] = dataset["y"][:id_split]
    elif split == "test":
        dataset["employee_salaries"] = dataset["employee_salaries"][id_split:]
        dataset["X"] = dataset["X"][id_split:]
        dataset["y"] = dataset["y"][id_split:]
    return dataset


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
        Size on disk: 36MB.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - ``medical_charge`` : pd.DataFrame, the dataframe. Shape: (163065, 12)
        - ``X`` : pd.DataFrame, features, i.e. the dataframe without the target
          labels. Shape: (163065, 11)
        - ``y`` : pd.DataFrame, target labels. Shape: (163065, 1)
        - ``metadata`` : a dictionary containing the name, description, source and
          target
    """
    return load_simple_dataset("medical_charge", data_home)


def fetch_midwest_survey(data_home=None):
    """Fetches the midwest survey dataset (classification), available at \
        https://github.com/skrub-data/skrub-data-files

    Description of the dataset:
        Survey to know if people self-identify as Midwesterners. Size on disk: 504KB.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - ``midwest_survey`` : pd.DataFrame, the dataframe. Shape: (2494, 29)
        - ``X`` : pd.DataFrame, features, i.e. the dataframe without the target
          labels. Shape: (2494, 28)
        - ``y`` : pd.DataFrame, target labels. Shape: (2494, 1)
        - ``metadata`` : a dictionary containing the name, description, source and
          target
    """
    return load_simple_dataset("midwest_survey", data_home)


def fetch_open_payments(data_home=None):
    """Fetches the open payments dataset (classification), available at \
        https://github.com/skrub-data/skrub-data-files

    Description of the dataset:
        Payments given by healthcare manufacturing companies to medical doctors
        or hospitals. Size on disk: 8.7MB.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - ``open_payments`` : pd.DataFrame, the dataframe. Shape: (73558, 6)
        - ``X`` : pd.DataFrame, features, i.e. the dataframe without the target
          labels. Shape: (73558, 5)
        - ``y`` : pd.DataFrame, target labels. Shape: (73558, 1)
        - ``metadata`` : a dictionary containing the name, description, source
          and target
    """
    return load_simple_dataset("open_payments", data_home)


def fetch_traffic_violations(data_home=None):
    """Fetches the traffic violations dataset (classification), available at \
        https://github.com/skrub-data/skrub-data-files

    Description of the dataset:
        This dataset contains traffic violation information from all electronic
        traffic violations issued in the Montgomery County, MD. Any information
        that can be used to uniquely identify the vehicle, the vehicle owner or
        the officer issuing the violation will not be published. Size on disk: 736MB.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - ``traffic_violations`` : pd.DataFrame, the dataframe. Shape: (1578154, 43)
        - ``X`` : pd.DataFrame, features, i.e. the dataframe without the target
          labels. Shape: (1578154, 42)
        - ``y`` : pd.DataFrame, target labels. Shape: (1578154, 1)
        - ``metadata`` : a dictionary containing the name, description, source and
          target
    """
    return load_simple_dataset("traffic_violations", data_home)


def fetch_drug_directory(data_home=None):
    """Fetches the drug directory dataset (classification), available at \
        https://github.com/skrub-data/skrub-data-files

    Description of the dataset:
        Product listing data submitted to the U.S. FDA for all unfinished,
        unapproved drugs. Size on disk: 44MB.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - ``drug_directory`` : pd.DataFrame, the dataframe. Shape: (120215, 21)
        - ``X`` : pd.DataFrame, features, i.e. the dataframe without the target
          labels. Shape: (120215, 20)
        - ``y`` : pd.DataFrame, target labels. Shape: (120215, 1)
        - ``metadata`` : a dictionary containing the name, description, source and
          target
    """
    return load_simple_dataset("drug_directory", data_home)


def fetch_credit_fraud(data_home=None, split="train"):
    """Fetch the credit fraud dataset (classification) available at \
        https://github.com/skrub-data/skrub-data-files

    This is an imbalanced binary classification use-case. This dataset consists of
    two tables:

    - baskets, containing the binary fraud target label
    - products

    Baskets contain at least one product each, so aggregation then joining operations
    are required to build a design matrix.
    Size on disk: 16MB.

    Parameters
    ----------
    data_home : str or path, default=None
        The directory where to download and unzip the files.

    split : str, default="train"
        The split to load. Can be either "train", "test", or "all".

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - ``baskets`` : pd.DataFrame, table containing baskets ID and target.
        Shape: (92790, 2)
        - ``product`` : pd.DataFrame, table containing features about products
          contained in baskets. Shape: (163357, 7)
        - ``metadata`` : a dictionary containing the name, description, source and
          target
    """
    if split not in ["train", "test", "all"]:
        raise ValueError(
            f"`split` must be one of ['train', 'test', 'all'], got: {split!r}."
        )
    dataset = load_dataset_files("credit_fraud", data_home)
    #  obtained by a quantile: dataset.baskets['ID'].quantile(.66)
    id_split = 76543  # noqa: F841
    if split == "train":
        dataset["baskets"] = dataset["baskets"].query("ID <= @id_split")
        dataset["products"] = dataset["products"].query("basket_ID <= @id_split")
    elif split == "test":
        dataset["baskets"] = dataset["baskets"].query("ID > @id_split")
        dataset["products"] = dataset["products"].query("basket_ID > @id_split")
    return dataset


def fetch_toxicity(data_home=None):
    """Fetch the toxicity dataset (classification) available at \
        https://github.com/skrub-data/skrub-data-files

    This is a balanced binary classification use-case, where the single table
    consists in only two columns:
    - `text`: the text of the comment
    - `is_toxic`: whether or not the comment is toxic
    Size on disk: 220KB.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - ``toxicity`` : pd.DataFrame, the dataframe. Shape: (1000, 2)
        - ``X`` : pd.DataFrame, features, i.e. the dataframe without the target
          labels. Shape: (1000, 1)
        - ``y`` : pd.DataFrame, target labels. Shape: (1000, 1)
        - ``metadata`` : a dictionary containing the name, description, source and
          target
    """
    return load_simple_dataset("toxicity", data_home)


def fetch_videogame_sales(data_home=None):
    """Fetch the videogame sales dataset (regression) available at \
        https://github.com/skrub-data/skrub-data-files

    This is a regression use-case, where the single table contains information
    about videogames such as the publisher and platform, and the goal is to
    predict the number of sales worldwide. Size on disk: 1.8MB.

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

        - ``videogame_sales`` : pd.DataFrame, the full dataframe. Shape: (16572, 11)
        - ``X`` : pd.DataFrame, features, i.e. the dataframe without the target
          labels. Shape: (16572, 5)
        - ``y`` : pd.DataFrame, target labels. Shape: (16572, 1)
        - ``metadata`` : a dictionary containing the name, source and target
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
    information. Size on disk: 1.3MB.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - ``bike_sharing``: pd.DataFrame, the full dataframe. Shape: (17379, 11)
        - ``X`` : pd.DataFrame, features, i.e. the dataframe without the target labels.
          Shape: (17379, 10)
        - ``y`` : pd.DataFrame, target labels. Shape: (17379, 1)
        - ``metadata`` : a dictionary containing the name and target
    """

    return load_simple_dataset("bike_sharing", data_home)


def fetch_movielens(data_home=None):
    """Fetch the movielens dataset (regression) available at \
        https://github.com/skrub-data/skrub-data-files

    This is a regression use-case, where the goal is to predict movie ratings.
    More details are provided in the output's ``metadata['description']``.
    Size on disk: 3.6MB.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - ``movies`` : pd.DataFrame, movie ID, title and genres. Shape: (9742, 3)
        - ``ratings``: pd.DataFrame, user ID, movie ID, rating. Shape: (100836, 4)
        - ``metadata`` : a dictionary containing the name source and description
    """

    return load_dataset_files("movielens", data_home)


def fetch_flight_delays(data_home=None):
    """Fetch the flight delays dataset (regression) available at \
        https://github.com/skrub-data/skrub-data-files

    This is a regression use-case, where the goal is to predict flight delays.
    Size on disk: 657MB.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - ``flights``: information about the flights, including departure and
          arrival airports, and delay. Shape: (2370030, 12)
        - ``airports``: information about airports, such as city and coordinates.
          The airport's ``iata`` can be matched to the flights' ``Origin`` and
          ``Dest``. Shape: (3376, 7)
        - ``weather``: weather data that could be used to help improve the delay
          predictions. Note the weather data is not measured at the airports
          directly but at weather stations, whose location and information is
          provided in ``stations``. Shape: (11282238, 5)
        - ``stations``: information about the weather stations. ``weather`` and
          ``stations`` can be joined on their ``ID`` columns. Weather stations
          can only be matched to the nearest airport based on the latitude and
          longitude. Shape: (124245, 9)
        - ``metadata`` : a dictionary containing the name  of the dataset.
    """
    return load_dataset_files("flight_delays", data_home)


def fetch_country_happiness(data_home=None):
    """Fetch the happiness index dataset (regression) available at \
        https://github.com/skrub-data/skrub-data-files

    This is a regression use-case, where the goal is to predict the happiness
    index. The dataset contains data from the `2022 World Happiness Report
    <https://worldhappiness.report/>`_, and from `the World Bank open data
    platform <https://data.worldbank.org/>`_. Size on disk: 64KB.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : sklearn.utils.Bunch
        A dictionary-like object with the following keys:

        - ``happiness_report``: dataframe, data from the world happiness report.
          Shape: (146, 12)
        - ``GDP_per_capita``: dataframe from the World Bank. Shape: (262, 2)
        - ``life_expectancy``: dataframe from the World Bank. Shape: (260, 2)
        - ``legal_rights_index``: dataframe from the World Bank. Shape: (238, 2)
        - ``metadata`` : a dictionary containing the name of the dataset, a
          description and the sources.
    """
    return load_dataset_files("country_happiness", data_home)
