import pytest
from benchmarks.utils import monitor, repr_func  # replace with the actual import path


def test_repr_func():
    def test_func(a, b, c=None, d=None):
        pass

    args = (1, 2)
    kwargs = {"c": 3, "d": 4}
    assert repr_func(test_func, args, kwargs) == "test_func(1, 2, c=3, d=4)"


def test_monitor_with_parameters():
    @monitor(parametrize={"a": [1, 2, 3], "b": [4, 5, 6]})
    def test_func(a, b):
        return {"output": a + b}

    df = test_func()
    assert len(df) == 9  # 3 values for a and 3 values for b, total 3*3=9 combinations
    assert "a" in df
    assert "b" in df
    assert "output" in df
    assert "call" in df
    assert "time" in df
    assert "memory" in df


def test_monitor_with_repeat():
    @monitor(repeat=5, parametrize={"a": [1, 2, 3]})
    def test_func(a):
        return {"output": a}

    df = test_func()
    assert len(df) == 15  # 3 values for a and 5 repetitions, total 3*5=15 combinations
    # check that each combination is repeated 5 times
    assert df["call"].value_counts().max() == 5
    assert df["call"].value_counts().min() == 5


def test_monitor_random_search():
    @monitor(parametrize={"a": [1, 2, 3, 4, 5]}, n_random_search=2)
    def test_func(a):
        return {"output": a}

    df = test_func()
    assert len(df) == 2  # only 2 random parameters
    # check that they are different
    assert df["call"][0] != df["call"][1]

    # with more parameters
    @monitor(
        parametrize={"a": [1, 2, 3, 4, 5], "b": [6, 7, 8, 9, 10]}, n_random_search=4
    )
    def test_func(a, b):
        return {"output": a + b}

    df = test_func()
    assert len(df) == 4  # only 4 random parameters
    assert df["call"][0] != df["call"][1] != df["call"][2] != df["call"][3]
