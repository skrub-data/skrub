import pytest

import skrub


def example_strings():
    return [
        skrub.var("a", "abc"),
        skrub.as_expr("abc"),
        skrub.var("a", "abc") + " def",
    ]


@pytest.mark.parametrize("a", example_strings())
def test_dir(a):
    assert "lower" in dir(a)
    assert "badattr" not in dir(a)


@pytest.mark.parametrize("a", example_strings())
def test_doc(a):
    assert "Create a new string object" in a.__doc__
