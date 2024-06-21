import re

import numpy as np
import pandas as pd
import pytest

from skrub import _dataframe as sbd
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


def test_check_no_column_name_duplicates_with_no_suffix(df_module):
    left = df_module.make_dataframe({"A": [], "B": []})
    right = df_module.make_dataframe({"C": []})
    _join_utils.check_column_name_duplicates(left, right, "")


def test_check_no_column_name_duplicates_after_adding_a_suffix(df_module):
    left = df_module.make_dataframe({"A": [], "B": []})
    right = df_module.make_dataframe({"B": []})
    _join_utils.check_column_name_duplicates(left, right, "_right")


def test_check_column_name_duplicates_after_adding_a_suffix(df_module):
    left = df_module.make_dataframe({"A": [], "B_right": []})
    right = df_module.make_dataframe({"B": []})
    with pytest.raises(ValueError, match=".*suffix '_right'.*['B_right']"):
        _join_utils.check_column_name_duplicates(left, right, "_right")


def test_add_column_name_suffix(df_module):
    df = df_module.make_dataframe({"one": [], "two three": [], "x": []})
    df = _join_utils.add_column_name_suffix(df, "")
    assert list(df.columns) == ["one", "two three", "x"]
    df = _join_utils.add_column_name_suffix(df, "_y")
    assert list(df.columns) == ["one_y", "two three_y", "x_y"]


@pytest.fixture
def left(df_module):
    return df_module.make_dataframe({"left_key": [1, 2, 2], "left_col": [10, 20, 30]})


def test_left_join_all_keys_in_right_dataframe(df_module, left):
    right = df_module.make_dataframe({"right_key": [2, 1], "right_col": ["b", "a"]})
    joined = _join_utils.left_join(
        left, right=right, left_on="left_key", right_on="right_key"
    )
    expected = df_module.make_dataframe(
        {
            "left_key": [1, 2, 2],
            "left_col": [10, 20, 30],
            "right_col": ["a", "b", "b"],
        }
    )
    df_module.assert_frame_equal(joined, expected)


def test_left_join_some_keys_not_in_right_dataframe(df_module, left):
    right = df_module.make_dataframe({"right_key": [2, 3], "right_col": ["a", "c"]})
    joined = _join_utils.left_join(
        left, right=right, left_on="left_key", right_on="right_key"
    )
    expected = df_module.make_dataframe(
        {
            "left_key": [1, 2, 2],
            "left_col": [10, 20, 30],
            "right_col": [np.nan, "a", "a"],
        }
    )
    df_module.assert_frame_equal(joined, expected)


def test_left_join_same_key_name(df_module, left):
    right = df_module.make_dataframe({"left_key": [2, 1], "right_col": ["b", "a"]})
    joined = _join_utils.left_join(
        left, right=right, left_on="left_key", right_on="left_key"
    )
    expected = df_module.make_dataframe(
        {
            "left_key": [1, 2, 2],
            "left_col": [10, 20, 30],
            "right_col": ["a", "b", "b"],
        }
    )
    df_module.assert_frame_equal(joined, expected)


def test_left_join_same_col_name(df_module, left):
    right = df_module.make_dataframe({"right_key": [2, 1], "left_col": ["b", "a"]})
    joined = _join_utils.left_join(
        left, right=right, left_on="left_key", right_on="right_key"
    )

    cols = sbd.column_names(joined)
    assert cols[:2] == ["left_key", "left_col"]
    assert re.match("left_col__skrub_.*__", cols[2]) is not None

    expected = df_module.make_dataframe(
        {
            "a": [1, 2, 2],
            "b": [10, 20, 30],
            "c": ["a", "b", "b"],
        }
    )
    # Renaming is necessary because a random tag has been added
    expected = sbd.set_column_names(expected, cols)
    df_module.assert_frame_equal(joined, expected)


def test_left_join_renaming_right_cols(df_module, left):
    right = df_module.make_dataframe({"right_key": [1, 2], "right_col": ["a", "b"]})
    joined = _join_utils.left_join(
        left,
        right=right,
        left_on="left_key",
        right_on="right_key",
        rename_right_cols="right.{}",
    )
    expected = df_module.make_dataframe(
        {
            "left_key": [1, 2, 2],
            "left_col": [10, 20, 30],
            "right.right_col": ["a", "b", "b"],
        }
    )
    df_module.assert_frame_equal(joined, expected)


def test_left_join_wrong_left_type(df_module):
    right = df_module.make_dataframe({"right_key": [1, 2], "right_col": ["a", "b"]})
    with pytest.raises(
        TypeError,
        match=(
            "`left` must be a pandas or polars dataframe, got <class 'numpy.ndarray'>."
        ),
    ):
        _join_utils.left_join(
            np.array([1, 2]), right=right, left_on="left_key", right_on="right_key"
        )


def test_left_join_wrong_right_type(df_module, left):
    with pytest.raises(
        TypeError,
        match=(
            "`right` must be a pandas or polars dataframe, got <class 'numpy.ndarray'>."
        ),
    ):
        _join_utils.left_join(
            left, right=np.array([1, 2]), left_on="left_key", right_on="right_key"
        )


def test_left_join_types_not_equal(df_module, left):
    try:
        import polars as pl
    except ImportError:
        pytest.skip(reason="Polars not available.")

    other_px = pd if df_module.module is pl else pl
    right = other_px.DataFrame(left)

    with pytest.raises(
        TypeError, match=r"`left` and `right` must be of the same dataframe type"
    ):
        _join_utils.left_join(
            left, right=right, left_on="left_key", right_on="right_key"
        )
