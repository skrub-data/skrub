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
