"""
Functions that generate example data.

"""

from __future__ import annotations

import string

import numpy as np
import pandas as pd
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
        List of duplicated examples with spelling mistakes
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
    bunch
        A dictionary-like object with the keys 'X', 'y' and 'orders'.
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
