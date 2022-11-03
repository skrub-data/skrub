import random

import numpy as np


def generate_data(n_samples, as_list=False):
    MAX_LIMIT = 255  # extended ASCII Character set
    i = 0
    str_list = []
    for i in range(n_samples):
        random_string = "category "
        for _ in range(100):
            random_integer = random.randint(0, MAX_LIMIT)
            random_string += chr(random_integer)
            if random_integer < 50:
                random_string += "  "
        i += 1
        str_list += [random_string]
    if as_list is True:
        X = str_list
    else:
        X = np.array(str_list).reshape(n_samples, 1)
    return X
