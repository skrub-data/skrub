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
    assert "skb" in dir(a)


def test_dir_no_preview():
    a = skrub.var("a")
    assert "skb" in dir(a)


@pytest.mark.parametrize("a", example_strings())
def test_doc(a):
    assert "Encode the string using the codec" in a.encode.__doc__


class _A:
    pass


def test_missing_doc():
    from skrub._expressions._expressions import _EXPR_INSTANCE_DOC

    assert skrub.X().__doc__ == _EXPR_INSTANCE_DOC
    assert skrub.X(_A()).__doc__ == _EXPR_INSTANCE_DOC


def test_expr_class_doc():
    from skrub._expressions._expressions import _EXPR_CLASS_DOC, Expr

    assert Expr.__doc__ == _EXPR_CLASS_DOC


@pytest.mark.parametrize("a", example_strings())
def test_signature(a):
    assert "encoding" in inspect.signature(a.encode).parameters


def test_missing_signature():
    with pytest.raises(AttributeError):
        skrub.X(0).__signature__


def test_key_completions():
    a = skrub.var("a", {"one": 1}) | skrub.var("b", {"two": 2})
    assert a._ipython_key_completions_() == ["one", "two"]
    assert skrub.X()._ipython_key_completions_() == []
    assert skrub.X(0)._ipython_key_completions_() == []


def test_repr_html():
    a = skrub.var("thename", "thevalue")
    r = a._repr_html_()
    if "Please install" in r:
        pytest.skip("graphviz not installed")
    assert "thename" in r and "thevalue" in r
    a = skrub.var("thename", skrub.toy_orders().orders)
    r = a._repr_html_()
    assert "thename" in r and "table-report" in r
    r = a["quantity"]._repr_html_()
    assert "quantity" in r and "table-report" in r and "product" not in r
    assert "thename" in skrub.var("thename")._repr_html_()
    # example without a name
    assert "add" in (skrub.var("thename", 0) + 2)._repr_html_()
    b = a.skb.apply_func(lambda x: x).skb.set_name("the name b")
    assert "the name b" in b._repr_html_()


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
    >>> skrub.as_expr({'a': 0})
    <Value dict>
    Result:
    ―――――――
    {'a': 0}
    >>> skrub.var('a', 1).skb.match({1: 10, 2: 20})
    <Match <Var 'a'>>
    Result:
    ―――――――
    10
    >>> from sklearn.preprocessing import StandardScaler, RobustScaler
    >>> skrub.X().skb.apply(StandardScaler())
    <Apply StandardScaler>
    >>> skrub.X().skb.apply('passthrough')
    <Apply 'passthrough'>
    >>> skrub.X().skb.apply(None)
    <Apply passthrough>
    >>> skrub.X().skb.apply(skrub.optional(StandardScaler(), name='scale'))
    <Apply StandardScaler>
    >>> skrub.X().skb.apply(
    ...     skrub.choose_from([RobustScaler(), StandardScaler()], name='scale'))
    <Apply RobustScaler>
    >>> skrub.as_expr({'a': 0})['a']
    <GetItem 'a'>
    Result:
    ―――――――
    0
    >>> skrub.as_expr({'a': 0, 'b': 1})[skrub.choose_from(['a', 'b'], name='c')]
    <GetItem choose_from(['a', 'b'], name='c')>
    Result:
    ―――――――
    0
    >>> skrub.as_expr({'a': 0, 'b': 1})[skrub.var('key', 'b')]
    <GetItem <Var 'key'>>
    Result:
    ―――――――
    1
    >>> skrub.as_expr('hello').upper()
    <CallMethod 'upper'>
    Result:
    ―――――――
    'HELLO'
    >>> a = skrub.var('a', 'hello')
    >>> b = skrub.var('b', 1)
    >>> skrub.as_expr({0: a.upper, 1: a.title})[b]()
    <Call "{ ... }[<Var 'b'>]">
    Result:
    ―――――――
    'Hello'
    >>> skrub.var('f', str.upper)('abc')
    <Call 'f'>
    Result:
    ―――――――
    'ABC'

    In cases that are hard to figure out we fall back on a less informative
    default

    >>> skrub.choose_from([str.upper, str.title], name='f').as_expr()('abc')
    <Call 'Value'>
    Result:
    ―――――――
    'ABC'
    >>> skrub.as_expr(str.upper)('abc')
    <Call 'Value'>
    Result:
    ―――――――
    'ABC'

    >>> a = skrub.var('a')
    >>> b = skrub.var('b')
    >>> c = skrub.var('c', 0)
    >>> a + b
    <BinOp: add>
    >>> - a
    <UnaryOp: neg>
    >>> 2 + a
    <BinOp: add>
    >>> c + c
    <BinOp: add>
    Result:
    ―――――――
    0
    >>> - c
    <UnaryOp: neg>
    Result:
    ―――――――
    0
    >>> 2 - c
    <BinOp: sub>
    Result:
    ―――――――
    2

    >>> X = skrub.X()
    >>> X.skb.concat_horizontal([X, X])
    <ConcatHorizontal: 3 dataframes>

    When we do not know the length of the list of dataframes to concatenate

    >>> X.skb.concat_horizontal(skrub.as_expr([X, X]))
    <ConcatHorizontal>

    if we end up applying a OnEachColumn, seeing the inner transformer is more
    informative.

    >>> from skrub._wrap_transformer import wrap_transformer
    >>> skrub.X().skb.apply(wrap_transformer(skrub.DatetimeEncoder(), ['a', 'b']))
    <Apply DatetimeEncoder>

    >>> skrub.choose_float(0.0, 10.0, n_steps=6, name='f')
    choose_float(0.0, 10.0, n_steps=6, name='f')

    >>> a = skrub.var('a', 0)
    >>> a
    <Var 'a'>
    Result:
    ―――――――
    0

    The shortened representation is used to show ``a`` in the match's repr:
    >>> a.skb.match({0: 10, 1: 11})
    <Match <Var 'a'>>
    Result:
    ―――――――
    10

    Misc objects that the user is less likely to see

    >>> skrub.X().skb
    <SkrubNamespace>
    >>> skrub.X()._skrub_impl.value
    NULL
    >>> print(skrub.X()._skrub_impl.value)
    NULL
    """


def test_format():
    assert f"{skrub.X()}" == "<Var 'X'>"
    with pytest.raises(ValueError, match="Invalid format specifier"):
        f"{skrub.X(0.2):.2f}"
