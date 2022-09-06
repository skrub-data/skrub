import pytest

from dirty_cat.utils import LRUDict, Version


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
