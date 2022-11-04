import random

import numpy as np


def generate_data(n: int = 100):
    MAX_LIMIT = 255  # extended ASCII Character set
    str_list = []
    for i in range(n):
        random_string = "aa"
        for _ in range(100):
            random_integer = random.randint(0, MAX_LIMIT)
            random_string += chr(random_integer)
            if random_integer < 50:
                random_string += "  "
        str_list += [random_string]
    return np.array(str_list).reshape(n, 1)
