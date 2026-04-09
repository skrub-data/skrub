import inspect

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier

import skrub


def example_strings():
    return [
        skrub.var("a", "abc"),
        skrub.as_data_op("abc"),
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
    from skrub._data_ops._data_ops import _DATA_OP_INSTANCE_DOC

    assert skrub.X().__doc__ == _DATA_OP_INSTANCE_DOC
    assert skrub.X(_A()).__doc__ == _DATA_OP_INSTANCE_DOC


def test_data_op_class_doc():
    from skrub._data_ops._data_ops import _DATA_OP_CLASS_DOC, DataOp

    assert DataOp.__doc__ == _DATA_OP_CLASS_DOC


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
    a = skrub.var("thename", skrub.datasets.toy_orders().orders)
    r = a._repr_html_()
    assert "thename" in r and "table-report" in r
    r = a["quantity"]._repr_html_()
    assert "quantity" in r and "table-report" in r and "product" not in r
    assert "thename" in skrub.var("thename")._repr_html_()
    # example without a name
    assert "add" in (skrub.var("thename", 0) + 2)._repr_html_()
    b = a.skb.apply_func(lambda x: x).skb.set_name("the name b")
    assert "the name b" in b._repr_html_()
    a = skrub.X(np.ones((5, 2))) + 10
    assert "on a subsample" not in a._repr_html_()
    a = skrub.X(np.ones((5, 2))).skb.subsample(n=2) + 10
    assert "on a subsample" in a._repr_html_()


def test_with_scoring_repr_html():
    X, y = make_classification(n_samples=10)
    r = (
        skrub.X(X)
        .skb.apply(DummyClassifier(), y=skrub.y(y))
        .skb.with_scoring("accuracy")
        ._repr_html_()
    )
    assert "accuracy" in r


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
    >>> skrub.as_data_op({'a': 0})
    <Value dict>
    Result:
    ―――――――
    {'a': 0}
    >>> skrub.var('a', 1).skb.match({1: 10, 2: 20})
    <Match <Var 'a'>>
    Result:
    ―――――――
    10
    >>> from sklearn.model_selection import KFold
    >>> skrub.var('df').skb.mark_as_X(cv=KFold(10), split_kwargs={})
    <X>
    >>> skrub.var('df').skb.mark_as_X(cv=KFold(10), split_kwargs={}).skb.set_name('design matrix')
    <design matrix | X>
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
    >>> skrub.as_data_op({'a': 0})['a']
    <GetItem 'a'>
    Result:
    ―――――――
    0
    >>> skrub.as_data_op({'a': 0, 'b': 1})[skrub.choose_from(['a', 'b'], name='c')]
    <GetItem choose_from(['a', 'b'], name='c')>
    Result:
    ―――――――
    0
    >>> skrub.as_data_op({'a': 0, 'b': 1})[skrub.var('key', 'b')]
    <GetItem <Var 'key'>>
    Result:
    ―――――――
    1
    >>> skrub.as_data_op('hello').upper()
    <CallMethod 'upper'>
    Result:
    ―――――――
    'HELLO'
    >>> a = skrub.var('a', 'hello')
    >>> b = skrub.var('b', 1)
    >>> skrub.as_data_op({0: a.upper, 1: a.title})[b]()
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

    >>> skrub.choose_from([str.upper, str.title], name='f').as_data_op()('abc')
    <Call 'Value'>
    Result:
    ―――――――
    'ABC'
    >>> skrub.as_data_op(str.upper)('abc')
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
    >>> X.skb.concat([X, X],axis=1)
    <Concat: 3 dataframes>
    >>> X.skb.concat([X, X],axis=0)
    <Concat: 3 dataframes>

    When we do not know the length of the list of dataframes to concatenate

    >>> X.skb.concat(skrub.as_data_op([X, X]),axis=0)
    <Concat>

    if we end up applying an ApplyToEachCol, seeing the inner transformer is more
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

    The preview indicates if subsampling took place:

    >>> import numpy as np

    >>> skrub.X(np.ones((5, 2))) + 10
    <BinOp: add>
    Result:
    ―――――――
    array([[11., 11.],
           [11., 11.],
           [11., 11.],
           [11., 11.],
           [11., 11.]])
    >>> skrub.X(np.ones((5, 2))).skb.subsample(n=2) + 10
    <BinOp: add>
    Result (on a subsample):
    ――――――――――――――――――――――――
    array([[11., 11.],
           [11., 11.]])


    short_repr of choices:

    >>> c1 = skrub.choose_float(10, 100)
    >>> c2 = skrub.choose_float(1, 100, log=True, n_steps=100, default=10)
    >>> e = skrub.var('x') + c1 + c2
    >>> print(e.skb.describe_param_grid())
    - choose_float(10, 100): choose_float(10, 100)
      choose_float(1, 100, log=True, n_...): choose_float(1, 100, log=True, n_steps=100, default=10)
    >>> from sklearn.dummy import DummyClassifier
    >>> from sklearn.metrics import get_scorer

    >>> data = skrub.var("df")
    >>> X = data[["description", "price"]].skb.mark_as_X(cv=2)
    >>> y = data["category"].skb.mark_as_y()
    >>> pred = X.skb.apply(DummyClassifier(), y=y)

    >>> sample_weight = X["price"]
    >>> pred.skb.with_scoring("accuracy", kwargs={"sample_weight": sample_weight})
    <Scoring <Apply DummyClassifier> (1 scorers)>
        This DataOp will be scored with:
          - 'accuracy'
        Use .skb.cross_validate(…) or .skb.make_learner(…).score(…) to compute scores.

    >>> pred.skb.with_scoring(["accuracy", get_scorer("roc_auc")]).skb.with_scoring(
    ...     "accuracy",
    ...     kwargs={"sample_weight": sample_weight},
    ...     name="weighted_accuracy",
    ... )
    <Scoring <Apply DummyClassifier> (3 scorers)>
        This DataOp will be scored with:
          - 'accuracy'
          - make_scorer(roc_auc_score, response_method=('decision_function', 'predict_proba'))
          - weighted_accuracy: 'accuracy'
        Use .skb.cross_validate(…) or .skb.make_learner(…).score(…) to compute scores.
    """  # noqa: E501


def test_format():
    assert f"{skrub.X()}" == "<Var 'X'>"
    with pytest.raises(ValueError, match="Invalid format specifier"):
        f"{skrub.X(0.2):.2f}"


def test_estimator_repr_html():
    # diagrams generated for scikit-learn estimators should contain the graph
    # drawing.

    data_op = (
        skrub.X()
        .skb.apply(skrub.SquashingScaler())
        .skb.apply(DummyClassifier(), y=skrub.y())
    )
    svg = data_op._repr_html_()

    learner = data_op.skb.make_learner()
    assert svg in learner._repr_html_()

    X, y = make_classification()
    learner.fit({"X": X, "y": y})
    assert svg in learner._repr_html_()

    search_repr = data_op.skb.make_randomized_search()._repr_html_()
    assert svg in search_repr
    assert "n_iter" in search_repr

    pytest.importorskip("optuna")
    optuna_search_repr = data_op.skb.make_randomized_search(
        backend="optuna"
    )._repr_html_()
    assert svg in optuna_search_repr
    assert "n_iter" in optuna_search_repr
