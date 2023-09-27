import pandas as pd
import pytest

from skrub import _join_utils


def test_add_column_name_suffix():
    dataframes = [
        pd.DataFrame(columns=["id", "col 1"]),
        pd.DataFrame(columns=["col 2"]),
        pd.DataFrame(),
    ]
    renamed = _join_utils.add_column_name_suffix(dataframes, "")
    assert pd.concat(dataframes).columns.tolist() == ["id", "col 1", "col 2"]
    assert pd.concat(renamed).columns.tolist() == ["id", "col 1", "col 2"]
    renamed = _join_utils.add_column_name_suffix(dataframes, "_x")
    assert pd.concat(dataframes).columns.tolist() == ["id", "col 1", "col 2"]
    assert pd.concat(renamed).columns.tolist() == ["id_x", "col 1_x", "col 2_x"]


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
