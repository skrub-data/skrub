import inspect

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
    assert "Encode the string using the codec" in a.encode.__doc__


@pytest.mark.parametrize("a", example_strings())
def test_signature(a):
    assert "encoding" in inspect.signature(a.encode).parameters


def test_key_completions():
    a = skrub.var("a", {"one": 1}) | skrub.var("b", {"two": 2})
    assert a._ipython_key_completions_() == ["one", "two"]
