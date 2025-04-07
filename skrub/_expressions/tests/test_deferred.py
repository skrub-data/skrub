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
@pytest.mark.parametrize("use_apply_func", [False, True])
def test_deferred_callables(func, use_apply_func):
    a = skrub.var("a", 10)
    if use_apply_func:
        b = a.skb.apply_func(func, 2.0)
    else:
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
    >>> a.skb.apply_func(F())
    <Call '<...F object at 0x...>'>
    Result:
    ―――――――
    20
    >>> a.skb.apply_func(lambda x: x * 2)
    <Call '<lambda>'>
    Result:
    ―――――――
    20
    >>> a.skb.apply_func(functools.partial(lambda x: x * 2))
    <Call 'functools.partial(<function <lambda> at 0x...>)'>
    Result:
    ―――――――
    20
    """


@pytest.mark.parametrize("use_apply_func", [False, True])
def test_deferred_default_value(use_apply_func):
    b = skrub.var("b")

    @skrub.deferred
    def f(a, b=b):
        return a + b

    a = skrub.var("a")
    if use_apply_func:
        c = a.skb.apply_func(f)
    else:
        c = f(a)
    assert c.skb.eval({"a": 1, "b": 2}) == 3


_B = skrub.var("b")


@pytest.mark.parametrize("use_apply_func", [False, True])
def test_deferred_global(use_apply_func):
    @skrub.deferred
    def f(a):
        return a + _B

    a = skrub.var("a")
    if use_apply_func:
        c = a.skb.apply_func(f)
    else:
        c = f(a)
    assert c.skb.eval({"a": 1, "b": 2}) == 3


@pytest.mark.parametrize("use_apply_func", [False, True])
def test_deferred_closure(use_apply_func):
    def g():
        b = skrub.var("b")

        @skrub.deferred
        def f(a):
            return a + b

        return f

    f = g()
    a = skrub.var("a")
    if use_apply_func:
        c = a.skb.apply_func(f)
    else:
        c = f(a)
    assert c.skb.eval({"a": 1, "b": 2}) == 3


@pytest.mark.parametrize("use_apply_func", [False, True])
def test_deferred_builtin(use_apply_func):
    if use_apply_func:
        a = skrub.as_expr("12").skb.apply_func(int)
    else:
        a = skrub.deferred(int)("12")
    assert a.skb.eval() == 12


@pytest.mark.parametrize("use_apply_func", [False, True])
def test_deferred_method(use_apply_func):
    if use_apply_func:
        a = skrub.as_expr("ABC").skb.apply_func(str.lower)
    else:
        a = skrub.deferred(str.lower)("ABC")
    assert a.skb.eval() == "abc"


def test_deferred_idempotent():
    @skrub.deferred
    def add_one(x):
        return x + 1

    a = skrub.var("a", 0)
    b = a.skb.apply_func(skrub.deferred(add_one))
    result = b.skb.eval()
    assert isinstance(result, int)
    assert result == 1


def test_deferred_expr():
    # applying deferred when func is already an expression
    a = skrub.var("a", "hello")
    b = skrub.deferred(a.upper)()
    result = b.skb.eval()
    assert isinstance(result, str)
    assert result == "HELLO"
