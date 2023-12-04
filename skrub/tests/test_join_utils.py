import pandas as pd
import pytest

from skrub import _join_utils


@pytest.mark.parametrize(
    ("main_key", "aux_key", "key"),
    [("a", None, "a"), ("a", "a", "a"), (None, "a", "a")],
)
def test_check_too_many_keys_errors(main_key, aux_key, key):
    with pytest.raises(ValueError, match="Can only pass"):
        _join_utils.check_key(main_key, aux_key, key)


@pytest.mark.parametrize(
    ("main_key", "aux_key"),
    [("a", None), (None, "a"), (None, None)],
)
def test_check_too_few_keys_errors(main_key, aux_key):
    with pytest.raises(ValueError, match="Must pass"):
        _join_utils.check_key(main_key, aux_key, None)


@pytest.mark.parametrize(
    ("main_key", "aux_key", "key", "result"),
    [
        ("a", ["b"], None, (["a"], ["b"])),
        (["a", "b"], ["A", "B"], None, (["a", "b"], ["A", "B"])),
        (None, None, ["a", "b"], (["a", "b"], ["a", "b"])),
        (None, None, "a", (["a"], ["a"])),
    ],
)
def test_check_key(main_key, aux_key, key, result):
    assert _join_utils.check_key(main_key, aux_key, key) == result


def test_check_key_length_mismatch():
    with pytest.raises(
        ValueError, match=r"'left' and 'right' keys.*different lengths \(1 and 2\)"
    ):
        _join_utils.check_key(
            "AB", ["A", "B"], None, main_key_name="left", aux_key_name="right"
        )


def test_check_column_name_duplicates():
    left = pd.DataFrame(columns=["A", "B"])
    right = pd.DataFrame(columns=["C"])
    _join_utils.check_column_name_duplicates(left, right, "")

    left = pd.DataFrame(columns=["A", "B"])
    right = pd.DataFrame(columns=["B"])
    _join_utils.check_column_name_duplicates(left, right, "_right")

    left = pd.DataFrame(columns=["A", "B_right"])
    right = pd.DataFrame(columns=["B"])
    with pytest.raises(ValueError, match=".*suffix '_right'.*['B_right']"):
        _join_utils.check_column_name_duplicates(left, right, "_right")

    left = pd.DataFrame(columns=["A", "A"])
    right = pd.DataFrame(columns=["B"])
    with pytest.raises(ValueError, match="Table 'left' has duplicate"):
        _join_utils.check_column_name_duplicates(
            left, right, "", main_table_name="left"
        )


def test_add_column_name_suffix():
    df = pd.DataFrame(columns=["one", "two three", "x"])
    df = _join_utils.add_column_name_suffix(df, "")
    assert list(df.columns) == ["one", "two three", "x"]
    df = _join_utils.add_column_name_suffix(df, "_y")
    assert list(df.columns) == ["one_y", "two three_y", "x_y"]
