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
    <Call '<F object at 0x71f27df29010>'>
    Result:
    ―――――――
    20
    >>> skrub.deferred(lambda x: x * 2)(a)
    <Call '<lambda>'>
    Result:
    ―――――――
    20
    >>> skrub.deferred(functools.partial(lambda x: x * 2))(a)
    <Call 'functools.partial(<function <lambda> at 0x71f24dfe79c0>)'>
    Result:
    ―――――――
    20
    """
