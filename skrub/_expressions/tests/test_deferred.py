import functools

import pytest

import skrub


class F:
    def __call__(self, x, factor=1.0):
        return x * factor


f_obj = F()


def f(x, factor=1.0):
    return x * factor


@pytest.mark.parametrize(
    "func", [f, f_obj, functools.partial(f), lambda x, factor: x * factor]
)
def test_deferred_callables(func):
    a = skrub.var("a", 10)
    b = skrub.deferred(func)(a, 2.0)
    assert b.skb.eval() == 20.0


def test_deferred_callables_repr():
    """
    >>> import skrub
    >>> import functools

    >>> class F:
    ...     def __call__(self, x):
    ...         return x * 2


    >>> a = skrub.var('a', 10)
    >>> skrub.deferred(F())(a)
    <Call '<...F object at 0x...>'>
    Result:
    ―――――――
    20
    >>> skrub.deferred(lambda x: x * 2)(a)
    <Call '<lambda>'>
    Result:
    ―――――――
    20
    >>> skrub.deferred(functools.partial(lambda x: x * 2))(a)
    <Call 'functools.partial(<function <lambda> at 0x...>)'>
    Result:
    ―――――――
    20
    """


def test_deferred_default_value():
    b = skrub.var("b")

    @skrub.deferred
    def f(a, b=b):
        return a + b

    a = skrub.var("a")
    c = f(a)
    assert c.skb.eval({"a": 1, "b": 2}) == 3


_B = skrub.var("b")


def test_deferred_global():
    @skrub.deferred
    def f(a):
        return a + _B

    a = skrub.var("a")
    c = f(a)
    assert c.skb.eval({"a": 1, "b": 2}) == 3


def test_deferred_closure():
    def g():
        b = skrub.var("b")

        @skrub.deferred
        def f(a):
            return a + b

        return f

    f = g()
    a = skrub.var("a")
    c = f(a)
    assert c.skb.eval({"a": 1, "b": 2}) == 3


def test_deferred_builtin():
    a = skrub.deferred(int)("12")
    assert a.skb.eval() == 12


def test_deferred_method():
    a = skrub.deferred(str.lower)("ABC")
    assert a.skb.eval() == "abc"
