import random

import numpy as np


def generate_data(
    n_samples,
    as_list=False,
    random_state: int | float | str | bytes | bytearray | None = None,
    sample_length: int = 100,
) -> np.ndarray:
    if random_state is not None:
        random.seed(random_state)
    MAX_LIMIT = 255  # extended ASCII Character set
    str_list = []
    for i in range(n_samples):
        random_string = "category "
        for _ in range(sample_length):
            random_integer = random.randint(1, MAX_LIMIT)
            random_string += chr(random_integer)
            if random_integer < 50:
                random_string += "  "
        str_list += [random_string]
    if as_list is True:
        X = str_list
    else:
        X = np.array(str_list).reshape(n_samples, 1)
    return X
