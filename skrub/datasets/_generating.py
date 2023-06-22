"""
Functions that generate example data.

"""
from __future__ import annotations

import string

import numpy as np
from sklearn.utils import check_random_state


def make_deduplication_data(
    examples: list[str],
    entries_per_example: list[int],
    prob_mistake_per_letter: float,
    random_state: int | np.random.RandomState | None = None,
) -> list[str]:
    """Duplicates examples with spelling mistakes.

    Characters are misspelled with probability `prob_mistake_per_letter`.

    Parameters
    ----------
    examples : list of str
        Examples to duplicate
    entries_per_example : list of int
        Number of duplications per example
    prob_mistake_per_letter : float in [0, 1]
        Probability of misspelling a character in duplications
    random_state : int, :obj:`~numpy.random.RandomState` instance, optional
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
            for i in np.random.choice(np.arange(26), len(mis_idx)).astype(int)
        ]
        # Introduce spelling mistakes at right examples and char locations per example
        str_as_list[mis_idx // len_ex, mis_idx % len_ex] = replacements
        # go back to 1d array of strings
        data.append(np.ascontiguousarray(str_as_list).view(f"U{len_ex}").ravel())
    return np.concatenate(data).tolist()
