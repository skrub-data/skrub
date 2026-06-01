"""
Functions that generate example data.

"""

from __future__ import annotations

import numbers
import string
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import Bunch, check_random_state


def make_deduplication_data(
    examples,
    entries_per_example,
    prob_mistake_per_letter=0.2,
    random_state=None,
):
    """Duplicates examples with spelling mistakes.

    Characters are misspelled with probability `prob_mistake_per_letter`.

    Parameters
    ----------
    examples : list of str
        Examples to duplicate.
    entries_per_example : list of int
        Number of duplications per example.
    prob_mistake_per_letter : float in [0, 1], default=0.2
        Probability of misspelling a character in duplications.
        By default, 1/5 of the characters will be misspelled.
    random_state : int, RandomState instance, optional
        Determines random number generation for dataset noise. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    list of str
        List of duplicated examples with spelling mistakes.

    Examples
    --------
    >>> from skrub.datasets import make_deduplication_data
    >>> make_deduplication_data(["string1", "string2"],
    ...                         entries_per_example=[4, 5],
    ...                         random_state=9)
    ['btrwng1', 'string1',
    'string1', 'string1',
    'saoing2', 'string2',
    'string2', 'string2',
    'string2']
    """
    rng = check_random_state(random_state)

    data = []
    for example, n_ex in zip(examples, entries_per_example):
        len_ex = len(example)
        # Generate a 2D array of chars of size (n_ex, len_ex)
        str_as_list = np.array([list(example)] * n_ex)
        # Randomly choose which characters are misspelled
        mis_idx = np.where(
            rng.random(len(example[0]) * n_ex) < prob_mistake_per_letter
        )[0]
        # Randomly pick with which character to replace
        replacements = [
            string.ascii_lowercase[i]
            for i in rng.choice(np.arange(26), len(mis_idx)).astype(int)
        ]
        # Introduce spelling mistakes at right examples and char locations per example
        str_as_list[mis_idx // len_ex, mis_idx % len_ex] = replacements
        # go back to 1d array of strings
        data.append(np.ascontiguousarray(str_as_list).view(f"U{len_ex}").ravel())
    return np.concatenate(data).tolist()


def toy_orders(split="train"):
    """Create a toy dataframe and corresponding targets for examples.

    Parameters
    ----------
    split : str, default="train"
        The split to load. Can be either "train", "test", or "all".

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        A dictionary-like object with the keys 'X', 'y' and 'orders'.

    Examples
    --------
    >>> from skrub.datasets import toy_orders
    >>> toy_orders(split="train").X
    ID	product	quantity	date
    0	1	pen	2	2020-04-03
    1	2	cup	3	2020-04-04
    2	3	cup	5	2020-04-04
    3	4	spoon	1	2020-04-05
    >>> toy_orders(split="train").y
    0    False
    1    False
    2     True
    3    False
    Name: delayed, dtype: bool

    If you want both X and y in a dataframe, use `.orders`:

    >>> toy_orders().orders
    ID	product	quantity	date	delayed
    0	1	pen	2	2020-04-03	False
    1	2	cup	3	2020-04-04	False
    2	3	cup	5	2020-04-04	True
    3	4	spoon	1	2020-04-05	False
    """
    X = pd.DataFrame(
        {
            "ID": [1, 2, 3, 4, 5, 6],
            "product": ["pen", "cup", "cup", "spoon", "cup", "fork"],
            "quantity": [2, 3, 5, 1, 5, 2],
            "date": [
                "2020-04-03",
                "2020-04-04",
                "2020-04-04",
                "2020-04-05",
                "2020-04-11",
                "2020-04-12",
            ],
        }
    )
    y = pd.Series([False, False, True, False, True, False], name="delayed")
    if split == "train":
        X, y = X.iloc[:4], y.iloc[:4]
    elif split == "test":
        X, y = X.iloc[4:], y.iloc[4:]
    else:
        assert split == "all", split
    return Bunch(X=X, y=y, orders=X.assign(delayed=y), orders_=X, delayed=y)


def toy_products():
    return pd.DataFrame(
        {
            "description": [
                "screen",
                "hammer",
                "keyboard",
                "usb key",
                "charger",
                "screwdriver",
            ],
            "price": [
                100,
                15,
                20,
                9,
                13,
                12,
            ],
            "seller": [
                "supermarket.com",
                "bestproducts.com",
                "supermarket.com",
                "bestproducts.com",
                "bestproducts.com",
                "supermarket.com",
            ],
            "category": [
                "electronics",
                "tools",
                "electronics",
                "electronics",
                "electronics",
                "tools",
            ],
        }
    )


def toy_cities(seed=0, size=1000, nulls=0.1, n_metrics=4):
    """Generate a synthetic dataframe example with a variety of column types.

    This can be used to showcase dataframes containing strings,
    dates and floats, columns containing null values, and strongly
    correlated columns.

    Contains the following columns:
    uid: A random identifying string of characters.
    cities: A city randomly picked in a list of 20, or a null value.
    encoded_cities: Ordinal encoding applied to the previous column.
    start: A datetime.
    end: A datetime later than the previous, or a null value.
    metric_1, metric_2, etc: Randomly chosen float values.

    Parameters
    ----------
    seed : int, default=0
        Seed for random generation.
    size : int, default=1000
        Number of rows in the output.
    nulls : float in [0, 1], default=0.1
        Probability of a cell in 'cities' or 'end' being null.
    n_metrics : int, default=4
        Number of 'metrics' columns added.

    Returns
    -------
    pandas dataframe
        The randomly-generated dataframe, with `size` rows and
        `5 + n_metrics` columns.

    Examples
    --------
    >>> from skrub.datasets import toy_cities
    >>> df = toy_cities(seed=5, size=3, n_metrics=2)
    >>> df # doctest: +SKIP
              uid     cities  ...  metric_0  metric_1
    0  IPbQyAGoYc  Stockholm  ...  0.227319  0.895448
    1  otDvgcachZ     Vienna  ...  0.872195  0.018517
    2  jHNmownYjU        NaN  ...  0.707496  0.001200
    """

    # Check that the nulls probability is valid
    if isinstance(nulls, bool) or not (isinstance(nulls, numbers.Number)):
        raise ValueError(f"nulls must be a number, got {nulls}.")
    elif not 0 <= nulls <= 1:
        raise ValueError(f"nulls must be a number between 0 and 1, got {nulls!r}.")

    # Check that the other variables are integers
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a positive integer, got {seed}.")
    if not isinstance(size, int) or size < 0:
        raise ValueError(f"size must be a positive integer, got {size}.")
    if not isinstance(n_metrics, int) or n_metrics < 0:
        raise ValueError(f"n_metrics must be a positive integer, got {n_metrics}.")

    rng = np.random.default_rng(seed=seed)
    now = datetime.fromisoformat("2024-01-01").timestamp()
    capitals = [
        "Amsterdam",
        "Athens",
        "Berlin",
        "Bucharest",
        "Budapest",
        "Copenhagen",
        "Dublin",
        "Helsinki",
        "London",
        "Madrid",
        "Paris",
        "Prague",
        "Riga",
        "Rome",
        "Sofia",
        "Stockholm",
        "Vienna",
        "Vilnius",
        "Warsaw",
        "Zagreb",
    ]

    # The first two columns are randomly constructed using lists.
    d = {}

    d["uid"] = [
        "".join(rng.choice(list(string.ascii_letters), 10)) for _ in range(size)
    ]
    d["cities"] = rng.choice(capitals, size=size)
    df_cities = pd.DataFrame(d)

    # `cities` gets assigned null values, and the ordinal encoder is run.
    p_cities = rng.uniform(0, 1, size=size)

    # Next, the "start" and "end" datetime columns are constructed.
    s = rng.integers(0, int(now), size=size)
    e = rng.integers(s, np.ones(size) * now)
    v = np.vstack([s, e])

    df_dates = pd.DataFrame(v.T, columns=["start", "end"])
    if hasattr(df_dates, "map"):
        df_dates = df_dates.map(datetime.fromtimestamp)
    else:
        df_dates = df_dates.applymap(datetime.fromtimestamp)
    # As above, "end" sees some of its values set to null.
    p_end = rng.uniform(0, 1, size=size)
    df_dates["end"] = df_dates["end"].where(p_end >= nulls)

    # Finally, constructing as many "metrics" float columns as specified.
    metric_cols = [f"metric_{k}" for k in range(n_metrics)]
    metrics_array = rng.random(size=(size, n_metrics))
    df_metrics = pd.DataFrame(metrics_array, columns=metric_cols)

    df_cities["cities"] = df_cities["cities"].where(p_cities >= nulls)
    df_cities["encoded_cities"] = OrdinalEncoder().fit_transform(df_cities[["cities"]])

    df = pd.concat((df_cities, df_dates, df_metrics), axis=1)

    return df
