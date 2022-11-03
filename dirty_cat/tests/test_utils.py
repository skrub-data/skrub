import random

import numpy as np
import pytest

from dirty_cat._utils import LRUDict, Version


def test_lrudict():
    dict_ = LRUDict(10)

    for x in range(15):
        dict_[x] = f"filled {x}"

    for x in range(5, 15):
        assert x in dict_
        assert dict_[x] == f"filled {x}"

    for x in range(5):
        assert x not in dict_


def test_version():
    # Test those specified in its docstring
    assert Version("1.5") <= Version("1.6.5")
    assert Version("1.5") <= "1.6.5"

    assert (Version("1-5", separator="-") == Version("1-6-5", separator="-")) is False
    assert (Version("1-5", separator="-") == "1-6-5") is False
    with pytest.raises(ValueError):
        assert not (Version("1-5", separator="-") == "1.6.5")

    # Test all comparison methods
    assert Version("1.0") == Version("1.0")
    assert (Version("1.0") != Version("1.0")) is False

    assert Version("1.0") < Version("1.1")
    assert Version("1.1") <= Version("1.5")
    assert Version("1.1") <= Version("1.1")
    assert (Version("1.1") < Version("1.1")) is False

    assert Version("1.1") > Version("0.5")
    assert Version("1.1") >= Version("0.9")
    assert Version("1.1") >= Version("1.1")
    assert (Version("1.1") >= Version("1.9")) is False


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
