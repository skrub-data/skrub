"""
Functions that generate example data.

"""

from __future__ import annotations

import string
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils import Bunch, check_random_state

from skrub._utils import random_string


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


def toy_random(seed=0, size=1000, nulls=0.1, n_metrics=4):
    np.random.seed(seed)
    t = time.time()
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

    d = {}

    d["uid"] = [random_string() for i in range(size)]
    d["cities"] = np.random.choice(capitals, size=size)
    for i in range(size):
        p = np.random.random()
        if p < nulls:
            d["cities"][i] = None

    cities_array = np.array(d["cities"]).reshape(-1, 1)
    d["encoded_cities"] = list(OrdinalEncoder().fit_transform(cities_array))

    start_times, end_times = [], []
    for _ in range(size):
        s = np.random.randint(0, int(t))
        e = np.random.randint(s, int(t))
        p = np.random.random()
        start_times.append(time.ctime(s))

        if p >= nulls:
            end_times.append(time.ctime(e))
        else:
            end_times.append(None)

    d["start_times"] = pd.to_datetime(start_times, format="mixed")
    d["end_times"] = pd.to_datetime(end_times, format="mixed")

    for k in range(n_metrics):
        d[f"metric_{k}"] = np.random.random(size)

    df = pd.DataFrame(d)

    df["encoded_cities"] = df.encoded_cities.explode().astype(int)

    return df
