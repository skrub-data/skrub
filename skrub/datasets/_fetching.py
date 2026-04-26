"""
Fetching functions to retrieve example datasets from GitHub and OSF.
"""

from pathlib import Path

from ._utils import download_dataset, load_dataset_files, load_simple_dataset


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
    data_home : str or path-like, default=None
        The directory where to download and unzip the files.

    split : str, default="all"
        The split to load. Can be either "train", "test", or "all".

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        A dictionary-like object with the following keys:

        employee_salaries : DataFrame of shape (9228, 8)
            The dataframe.
        X : DataFrame of shape (9228, 7)
            Features, i.e. the dataframe without the target labels.
        y : DataFrame of shape (9228, 1)
            Target labels.
        metadata : dict
            A dictionary containing the name, description, source and target.
        path : str
            The path to the employee salaries CSV file.
    """
    if split not in ["train", "test", "all"]:
        raise ValueError(
            f"`split` must be one of ['train', 'test', 'all'], got: {split!r}."
        )
    dataset = load_simple_dataset("employee_salaries", data_home)

    id_split = 8000
    if split == "train":
        dataset["employee_salaries"] = dataset["employee_salaries"][:id_split]
        train_path = Path(dataset["employee_salaries_path"]).with_name(
            "employee_salaries_train.csv"
        )
        dataset["employee_salaries_path"] = str(train_path)
        dataset["path"] = str(train_path)
        dataset["employee_salaries"].to_csv(str(train_path), index=False)
        dataset["X"] = dataset["X"][:id_split]
        dataset["y"] = dataset["y"][:id_split]
    elif split == "test":
        dataset["employee_salaries"] = dataset["employee_salaries"][id_split:]
        test_path = Path(dataset["employee_salaries_path"]).with_name(
            "employee_salaries_test.csv"
        )
        dataset["employee_salaries_path"] = str(test_path)
        dataset["path"] = str(test_path)
        dataset["employee_salaries"].to_csv(str(test_path), index=False)
        dataset["X"] = dataset["X"][id_split:]
        dataset["y"] = dataset["y"][id_split:]
    return dataset


def fetch_medical_charge(data_home=None):
    """Fetches the medical charge dataset (regression), available at \
        https://github.com/skrub-data/skrub-data-files

    Description of the dataset:
        The dataset provides information on inpatient discharges for Medicare
        fee-for-service beneficiaries. It includes information
        on utilization, payment (total payment and Medicare payment), and
        hospital-specific charges for the more than 3,000 U.S. hospitals that
        receive Medicare Inpatient Prospective Payment System (IPPS) payments.
        Size on disk: 36MB.

    Parameters
    ----------
    data_home : str or path-like, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        A dictionary-like object with the following keys:

        medical_charge : DataFrame of shape (163065, 12)
            The dataframe.
        X : DataFrame of shape (163065, 11)
            Features, i.e. the dataframe without the target labels.
        y : DataFrame of shape (163065, 1)
            Target labels.
        metadata : dict
            A dictionary containing the name, description, source and target.
        path : str
            The path to the medical charge CSV file.
    """
    return load_simple_dataset("medical_charge", data_home)


def fetch_midwest_survey(data_home=None):
    """Fetches the midwest survey dataset (classification), available at \
        https://github.com/skrub-data/skrub-data-files

    Description of the dataset:
        Survey to know if people self-identify as Midwesterners. Size on disk: 504KB.

    Parameters
    ----------
    data_home : str or path-like, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        A dictionary-like object with the following keys:

        midwest_survey : DataFrame of shape (2494, 29)
            The dataframe.
        X : DataFrame of shape (2494, 28)
            Features, i.e. the dataframe without the target labels.
        y : DataFrame of shape (2494, 1)
            Target labels,
        metadata : dict
            A dictionary containing the name, description, source and target.
        path : str
            The path to the midwest survey CSV file.
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
    data_home : str or path-like, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        A dictionary-like object with the following keys:

        open_payments : DataFrame of shape (73558, 6)
            The dataframe.
        X : DataFrame of shape (73558, 5)
            Features, i.e. the dataframe without the target labels.
        y : DataFrame of shape (73558, 1)
            Target labels.
        metadata : dict
            A dictionary containing the name, description, source and target.
        path : str
            The path to the open payments CSV file.
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
    data_home : str or path-like, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        A dictionary-like object with the following keys:

        traffic_violations : DataFrame of shape (1578154, 43)
            The dataframe.
        X : DataFrame of shape (1578154, 42)
            Features, i.e. the dataframe without the target labels.
        y : DataFrame of shape (1578154, 1)
            Target labels.
        metadata : dict
            A dictionary containing the name, description, source and target.
        path : str
            The path to the traffic violations CSV file.
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
    data_home : str or path-like, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        A dictionary-like object with the following keys:

        drug_directory : DataFrame of shape (120215, 21)
            The dataframe.
        X : DataFrame of shape (120215, 20)
            Features, i.e. the dataframe without the target labels.
        y : DataFrame of shape (120215, 1)
            The target labels.
        metadata : dict
            A dictionary containing the name, description, source and target.
        path : str
            The path to the drug directory CSV file.
    """
    return load_simple_dataset("drug_directory", data_home)


def fetch_credit_fraud(data_home=None, split="train"):
    """Fetch the credit fraud dataset (classification).

    Available at https://github.com/skrub-data/skrub-data-files

    This is an imbalanced binary classification use-case. This dataset consists of
    two tables:

    - baskets, containing the binary fraud target label
    - products

    Baskets contain at least one product each, so aggregation then joining operations
    are required to build a design matrix.
    Size on disk: 16MB.

    Parameters
    ----------
    data_home : str or path-like, default=None
        The directory where to download and unzip the files.

    split : str, default="train"
        The split to load. Can be either "train", "test", or "all".

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        A dictionary-like object with the following keys:

        baskets : DataFrame of shape (92790, 2)
            Table containing baskets ID and target.
        products : DataFrame of shape (163357, 7)
            Table containing features about products contained in baskets
        metadata : dict
            A dictionary containing the name, description, source and target.
        baskets_path : str
            The path to the baskets CSV file.
        products_path : str
            The path to the products CSV file.
    """
    if split not in ["train", "test", "all"]:
        raise ValueError(
            f"`split` must be one of ['train', 'test', 'all'], got: {split!r}."
        )
    dataset = load_dataset_files("credit_fraud", data_home)
    #  obtained by a quantile: dataset.baskets['ID'].quantile(.66)
    id_split = 76543  # noqa
    if split == "train":
        dataset["baskets"] = dataset["baskets"].query("ID <= @id_split")
        dataset["products"] = dataset["products"].query("basket_ID <= @id_split")

        train_path = Path(dataset["baskets_path"]).with_name("baskets_train.csv")
        dataset["baskets_path"] = str(train_path)
        dataset["baskets"].to_csv(str(train_path), index=False)

        train_path = Path(dataset["products_path"]).with_name("products_train.csv")
        dataset["products_path"] = str(train_path)
        dataset["products"].to_csv(str(train_path), index=False)
    elif split == "test":
        dataset["baskets"] = dataset["baskets"].query("ID > @id_split")
        dataset["products"] = dataset["products"].query("basket_ID > @id_split")

        test_path = Path(dataset["baskets_path"]).with_name("baskets_test.csv")
        dataset["baskets_path"] = str(test_path)
        dataset["baskets"].to_csv(str(test_path), index=False)

        test_path = Path(dataset["products_path"]).with_name("products_test.csv")
        dataset["products_path"] = str(test_path)
        dataset["products"].to_csv(str(test_path), index=False)
    return dataset


def fetch_toxicity(data_home=None):
    """Fetch the toxicity dataset (classification) available at \
        https://github.com/skrub-data/skrub-data-files

    This is a balanced binary classification use-case, where the single table
    consists in only two columns:

   - ``text``: the text of the comment
   - ``is_toxic``: whether or not the comment is toxic

    Size on disk: 220KB.

    Parameters
    ----------
    data_home : str or path-like, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        A dictionary-like object with the following keys:

        toxicity : DataFrame of shape (1000, 2)
            The dataframe.
        X : DataFrame of shape (1000, 1)
            Features, i.e. the dataframe without the target.
        y : DataFrame of shape (1000, 1)
            Target labels.
        metadata : dict
            A dictionary containing the name, description, source and target.
        path : str
            The path to the toxicity CSV file.
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
    data_home : str or path-like, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        A dictionary-like object with the following keys:

        videogame_sales : DataFrame of shape (16572, 11)
            The dataframe.
        X : DataFrame of shape (16572, 5)
            Features, i.e. the dataframe without the target labels.
        y : DataFrame of shape (16572, 1)
            Target labels.
        metadata : dict
            A dictionary containing the name, source and target.
        path : str
            The path to the videogame sales CSV file.
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
    data_home : str or path-like, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        A dictionary-like object with the following keys:

        bike_sharing : DataFrame of shape (17379, 11)
            The full dataframe.
        X : DataFrame of shape (17379, 10)
            Features, i.e. the dataframe without the target labels.
        y : DataFrame of shape (17379, 1)
            Target labels.
        metadata : dict
            A dictionary containing the name and target.
        path : str
            The path to the bike sharing CSV file.
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
    data_home : str or path-like, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        A dictionary-like object with the following keys:

        movies : DataFrame of shape (9742, 3)
            Dataframe with movie titles and genres.
        ratings : DataFrame of shape (100836, 4)
            Dataframe with ratings of movies.
        metadata : dict
            A dictionary containing the name source and description.
        movies_path : str
            The path to the movies CSV file.
        ratings_path : str
            The path to the ratings CSV file.
    """

    return load_dataset_files("movielens", data_home)


def fetch_flight_delays(data_home=None):
    """Fetch the flight delays dataset (regression) available at \
        https://github.com/skrub-data/skrub-data-files

    This is a regression use-case, where the goal is to predict flight delays.
    Size on disk: 657MB.

    Parameters
    ----------
    data_home : str or path-like, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        A dictionary-like object with the following keys:

        flights : DataFrame of shape (2370030, 12)
            Information about the flights, including departure and
            arrival airports, and delay.
        airports : DataFrame of shape (3376, 7)
            Information about airports, such as city and coordinates.
            The airport's ``iata`` can be matched to the flights' ``Origin`` and
            ``Dest``.
        weather : DataFrame of shape (11282238, 5)
            Weather data that could be used to help improve the delay
            predictions. Note the weather data is not measured at the airports
            directly but at weather stations, whose location and information is
            provided in ``stations``.
        stations : dataframe of shape (124245, 9)
            Information about the weather stations. ``weather`` and
            ``stations`` can be joined on their ``ID`` columns. Weather stations
            can only be matched to the nearest airport based on the latitude and
            longitude.
        metadata : dict
            A dictionary containing the name of the dataset.
        flights_path : str
            The path to the flights CSV file.
        airports_path : str
            The path to the airports CSV file.
        weather_path : str
            The path to the weather CSV file.
        stations_path : str
            The path to the stations CSV file.
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
    data_home : str or path-like, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        A dictionary-like object with the following keys:

        happiness_report : DataFrame of shape (146, 12)
            Data from the world happiness report.
        GDP_per_capita : DataFrame of shape (262, 2)
            Data from the World Bank.
        life_expectancy : DataFrame of shape (260, 2)
            Data from the World Bank.
        legal_rights_index : DataFrame of shape (238, 2)
            Data from the World Bank.
        metadata : dict
            A dictionary containing the name of the dataset, a description
            and the sources.
        happiness_report_path : str
            The path to the happiness report CSV file.
        GDP_per_capita_path : str
            The path to the GDP per capita CSV file.
        life_expectancy_path : str
            The path to the life expectancy CSV file.
        legal_rights_index_path : str
            The path to the legal rights index CSV file.
    """
    return load_dataset_files("country_happiness", data_home)


def fetch_california_housing(data_home=None):
    """Fetches the california housing dataset (regression), available at \
        https://github.com/skrub-data/skrub-data-files

    Description of the dataset:
        This dataset was obtained from the StatLib repository:
        https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html

        The target variable is the median house value for California districts,
        expressed in hundreds of thousands of dollars ($100,000).

        This dataset was derived from the 1990 U.S. census, using one row per census
        block group. A block group is the smallest geographical unit for which the U.S.
        Census Bureau publishes sample data (a block group typically has a population of
        600 to 3,000 people).

        A household is a group of people residing within a home. Since the average
        number of rooms and bedrooms in this dataset are provided per household, these
        columns may take surprisingly large values for block groups with few households
        and many empty houses, such as vacation resorts.

        It can be downloaded/loaded using the sklearn.datasets.fetch_california_housing
        function.
        Size on disk: 1.80MB.

    Parameters
    ----------
    data_home : str or path-like, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        A dictionary-like object with the following keys:

        california_housing : DataFrame of shape (20640, 9)
            A dataframe with the California housing data.
        X : DataFrame of shape (20640, 8)
            Features, i.e. the dataframe without the target labels.
        y : DataFrame of shape (20640, 1)
            Target labels.
        metadata : dict
            A dictionary containing the name, description, source and target.
        path : str
            The path to the california housing CSV file.
    """
    return load_simple_dataset("california_housing", data_home)


def fetch_electricity_usage(data_home=None):
    """Fetches the electricity usage dataset (forecasting), available at \
        https://github.com/skrub-data/skrub-data-files

    Description of the dataset:
        This dataset was generated from data obtained from the
        ENTSOE Open Data portal.:
        https://transparency.entsoe.eu/load-domain/r2/totalLoadR2/show?name=&defaultValue=false&viewType=TABLE&areaType=BZN&atch=false&dateTime.dateTime=12.10.2023+00:00%7CUTC%7CDAY&biddingZone.values=CTY%7C10YFR-RTE------C!BZN%7C10YFR-RTE------C&dateTime.timezone=UTC&dateTime.timezone_input=UTC#
        and the Open Meteo Historcal Weather API:
        https://open-meteo.com/en/docs/historical-forecast-api
        This is a time-series forecasting use case.

        This dataset gives the total electricity load in MW in France,
        covering a time range from March 23, 2021 to May 31,
        2025.

        In addition, the dataset contains weather data for several cities
        within France.

        It can be downloaded/loaded using the
        sklearn.datasets.fetch_electricity_usage function.
        Size on disk: 26MB.

    Parameters
    ----------
    data_home: str or path, default=None
        The directory where to download and unzip the files.

    Returns
    -------
    Path : PosixPath
         The path to the electricity usage CSV files

    """
    return download_dataset("electricity_usage", data_home=None)
