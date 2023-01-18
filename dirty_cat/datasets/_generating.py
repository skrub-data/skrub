"""
Functions that generate example data.

"""
from __future__ import annotations

from typing import List, Optional, Union
import string

import numpy as np

from sklearn.utils import check_random_state

def make_deduplication_data(
    examples: List[str],
    entries_per_example: List[int],
    prob_mistake_per_letter: float,
    random_state: Optional[Union[int, np.random.RandomState]] = None,
) -> List[str]:
    """Duplicates examples with spelling mistakes.
    Characters are misspelled with probability `prob_mistake_per_letter`.

    Parameters
    ----------
    examples : List[str]
        examples to duplicate
    entries_per_example : List[int]
        number of duplications per example
    prob_mistake_per_letter : float in [0, 1]
        probability of misspelling a character in duplications
    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset noise. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    List[str]
        list of duplicated examples with spelling mistakes
    """
    rng = check_random_state(random_state)

    data = []
    for example, n_ex in zip(examples, entries_per_example):
        len_ex = len(example)
        # generate a 2D array of chars of size (n_ex, len_ex)
        str_as_list = np.array([list(example)] * n_ex)
        # randomly choose which characters are misspelled
        idxes = np.where(
            rng.random(len(example[0]) * n_ex) < prob_mistake_per_letter
        )[0]
        # and randomly pick with which character to replace
        replacements = [
            string.ascii_lowercase[i]
            for i in np.random.choice(np.arange(26), len(idxes)).astype(int)
        ]
        # introduce spelling mistakes at right examples and char locations per example
        str_as_list[idxes // len_ex, idxes % len_ex] = replacements
        # go back to 1d array of strings
        data.append(np.ascontiguousarray(str_as_list).view(f"U{len_ex}").ravel())
    return np.concatenate(data).tolist()
