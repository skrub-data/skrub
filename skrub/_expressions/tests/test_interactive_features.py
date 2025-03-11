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


def test_repr_html():
    a = skrub.var("thename", "thevalue")
    r = a._repr_html_()
    assert "thename" in r and "thevalue" in r
    a = skrub.var("thename", skrub.toy_orders().orders)
    r = a._repr_html_()
    assert "thename" in r and "table-report" in r


def test_repr():
    r"""
    >>> import skrub
    >>> a = skrub.var('a', 'one') + ' two'
    >>> a
    <BinOp: add>
    Result:
    ―――――――
    'one two'
    >>> f'a = {a}'
    'a = <BinOp: add>'
    >>> print(f'a:\n{a:preview}')
    a:
    <BinOp: add>
    Result:
    ―――――――
    'one two'
    """
