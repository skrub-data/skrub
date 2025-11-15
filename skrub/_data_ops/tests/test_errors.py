import pickle
import re
import traceback

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import skrub
from skrub._utils import PassThrough

#
# Using eager statements on DataOps
#


def test_for():
    a = skrub.var("a", [1, 2, 3])
    with pytest.raises(
        TypeError, match=".*it is not possible to eagerly iterate over it"
    ):
        for _ in a:
            pass


def test_if():
    a = skrub.var("a", True)
    with pytest.raises(
        TypeError, match=".*it is not possible to eagerly use its Boolean value"
    ):
        if a:
            pass


def test_contains():
    a = skrub.var("a", [1, 2, 3])
    with pytest.raises(
        TypeError, match=".*it is not possible to eagerly perform membership tests"
    ):
        2 in a  # noqa: B015


def test_setitem():
    a = skrub.var("a", {})
    with pytest.raises(TypeError, match="Do not modify a DataOp in-place"):
        a["one"] = 1
    a = skrub.var("a", skrub.datasets.toy_orders().orders)
    with pytest.raises(
        TypeError, match=r"(?s)Do not modify a DataOp in-place.*df = df\.assign"
    ):
        a["one"] = 1


def test_setattr():
    class A:
        pass

    a = skrub.var("a", A())
    with pytest.raises(TypeError, match="Do not modify a DataOp in-place"):
        a.b = 0


def test_func_returning_none():
    a = skrub.var("a", [])
    with pytest.warns(UserWarning, match=r"Calling '\.append\(\)' returned None"):
        out = a.append(0)
    assert out.skb.eval() is None


#
# Check that evaluation failures are caught during preview if possible
#


def test_preview_failure():
    # ok: we don't have data so cannot evaluate
    skrub.X() / 0
    # catch the zero division early
    with pytest.raises(
        Exception, match=r"(?s)'__truediv__\(\)' failed.*ZeroDivisionError"
    ):
        skrub.X(1) / 0


#
# pickling
#


class NoPickle:
    _msg = "cannot pickle NoPickle"

    def __deepcopy__(self, mem):
        return self

    def __getstate__(self):
        raise pickle.PicklingError(self._msg)


class NoPickleRecursion(NoPickle):
    _msg = "cannot pickle NoPickleRecursion something about recursion"


def _pickle_msg_pattern(cls):
    pattern = r"The check to verify that the learner can be serialized failed\."
    if cls is not NoPickleRecursion:
        return pattern
    return pattern + " Is a step in the learner holding a reference to the full learner"


@pytest.mark.parametrize("cls", [NoPickle, NoPickleRecursion])
def test_pickling_preview_failure(cls):
    with pytest.raises(
        pickle.PicklingError,
        match=_pickle_msg_pattern(cls),
    ):
        (skrub.X([]) + [cls()]).skb.make_learner()


@pytest.mark.parametrize("cls", [NoPickle, NoPickleRecursion])
def test_pickling_estimator_failure(cls):
    a = []
    e = skrub.X([]) + a
    a.append(cls())
    with pytest.raises(
        pickle.PicklingError,
        match=_pickle_msg_pattern(cls),
    ):
        e.skb.make_learner()


#
# Choices, X and y
#


def test_duplicate_choice_name():
    with pytest.raises(
        ValueError, match=r".*The name 'b' was used for 2 different objects"
    ):
        (
            skrub.var("a")
            + skrub.choose_from([1, 2], name="b")
            + skrub.choose_int(0, 4, name="b")
        )

    with pytest.raises(
        ValueError, match=r".*The name 'b' was used for 2 different objects"
    ):
        (
            skrub.var("a")
            + skrub.var("b")
            + (skrub.var("c") + skrub.var("d")).skb.set_name("b")
        )

    with pytest.raises(
        ValueError, match=r".*The name 'a' was used for 2 different objects"
    ):
        (skrub.var("a") + skrub.choose_from([1, 2], name="a"))

    with pytest.raises(
        ValueError, match=r".*The name 'X' was used for 2 different objects"
    ):
        skrub.X() + skrub.var("X")

    with pytest.raises(
        ValueError, match=r"(?s).*2 different objects.*Jupyter notebook cell"
    ):
        skrub.as_data_op([skrub.var("a"), skrub.var("a")])


def test_duplicate_X():
    with pytest.raises(
        ValueError, match=r"Only one node can be marked with `mark_as_X\(\)`"
    ):
        skrub.X() + skrub.var("a").skb.mark_as_X()


def test_duplicate_y():
    with pytest.raises(
        ValueError, match=r"Only one node can be marked with `mark_as_y\(\)`"
    ):
        skrub.y() + skrub.var("a").skb.mark_as_y()


def test_missing_X_or_y():
    X_a, y_a = make_classification(random_state=0)
    env = {"X": X_a, "y": y_a}

    X, y = skrub.var("X"), skrub.var("y")
    with pytest.raises(
        ValueError, match=r'DataOp should have a node marked with "mark_as_X\(\)"'
    ):
        X.skb.apply(LogisticRegression(), y=y.skb.mark_as_y()).skb.cross_validate(env)
    with pytest.raises(
        ValueError, match=r'DataOp should have a node marked with "mark_as_y\(\)"'
    ):
        X.skb.mark_as_X().skb.apply(LogisticRegression(), y=y).skb.cross_validate(env)
    # now both are correctly marked:
    X.skb.mark_as_X().skb.apply(
        LogisticRegression(), y=y.skb.mark_as_y()
    ).skb.cross_validate(env)


def test_warn_if_choice_before_X_or_y():
    X_a, y_a = make_classification(random_state=0)

    # A choice appears before X
    with pytest.warns(
        UserWarning,
        match=r"The following choices are used in the construction of X or y"
        r".*\[choose_bool\(name='with_mean'\)\]",
    ):
        skrub.var("X", X_a).skb.apply(
            StandardScaler(**skrub.choose_bool(name="with_mean"))
        ).skb.mark_as_X().skb.apply(
            DummyClassifier(
                **skrub.choose_from(["prior", "most_frequent"], name="strategy")
            ),
            y=(skrub.y(y_a) + skrub.choose_from([1, -1], name="z")),
        )
    # A choice appears before y
    with pytest.warns(
        UserWarning,
        match=r"The following choices are used in the construction of X or y"
        r".*\[choose_from\(\[1, -1\], name='z'\)\]",
    ):
        skrub.X(X_a).skb.apply(
            StandardScaler(**skrub.choose_bool(name="with_mean"))
        ).skb.apply(
            DummyClassifier(
                **skrub.choose_from(["prior", "most_frequent"], name="strategy")
            ),
            y=(
                skrub.var("y", y_a) + skrub.choose_from([1, -1], name="z")
            ).skb.mark_as_y(),
        )


def test_inconsistent_subsampling():
    X_a, y_a = make_classification(random_state=0)

    # X subsampled but not y
    X, y = skrub.var("X", X_a).skb.subsample(n=10), skrub.var("y", y_a)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "`X` was subsampled with `.skb.subsample()` but `y` was not, resulting in"
            " different sizes (X: 10, y: 100)"
        ),
    ):
        X.skb.apply(LogisticRegression(), y=y)

    # y subsampled but not X
    X, y = skrub.var("X", X_a), skrub.var("y", y_a).skb.subsample(n=10)
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "`y` was subsampled with `.skb.subsample()` but `X` was not, resulting in"
            " different sizes (X: 100, y: 10)"
        ),
    ):
        X.skb.apply(LogisticRegression(), y=y)

    # Same without previews
    X, y = skrub.var("X"), skrub.var("y").skb.subsample(n=10)
    pred = X.skb.apply(LogisticRegression(), y=y)
    # evaluating without subsampling works
    pred.skb.eval({"X": X_a, "y": y_a})
    # evaluating with subsampling raises
    with pytest.raises(
        (ValueError, RuntimeError),
        match=(
            "(`y` was subsampled.*`X` was not)|(Evaluation of node <Apply"
            " LogisticRegression> failed)"
        ),
    ):
        pred.skb.eval({"X": X_a, "y": y_a}, keep_subsampling=True)

    # In the following case subsampling is consistent but the shapes don't
    # match for some other reason
    msg = re.escape(
        "Found input variables with inconsistent numbers of samples: X: 99, y: 100"
    )
    X = skrub.var("X", X_a[:-1]).skb.subsample(n=1000)
    y = skrub.var("y", y_a).skb.subsample(n=1000)
    with pytest.raises(RuntimeError, match=msg):
        X.skb.apply(LogisticRegression(), y=y)
    X = skrub.var("X", X_a[:-1])
    y = skrub.var("y", y_a)
    with pytest.raises(RuntimeError, match=msg):
        X.skb.apply(LogisticRegression(), y=y)


def test_inconsistent_subsampling_and_split_order():
    X_a, y_a = make_classification(random_state=0)

    # Both are subsampled but X is subsampled after splitting and y is subsampled before
    X = skrub.var("X", X_a).skb.mark_as_X().skb.subsample(n=10)
    y = skrub.var("y", y_a).skb.subsample(n=10).skb.mark_as_y()
    pred = X.skb.apply(LogisticRegression(), y=y)

    # computing preview and eval works as there is no splitting
    assert pred.shape.skb.preview() == (10,)
    assert pred.shape.skb.eval() == (100,)
    msg = re.escape(
        "`y` was subsampled with `.skb.subsample()` but `X` was not, resulting in"
        " different sizes (X: 100, y: 10)."
        "\nAre `.skb.subsample()` and `.skb.mark_as_*()` applied in the same order"
        " for both X and y?"
    )
    # splitting with subsampling disabled works
    pred.skb.train_test_split()
    # but splitting with subsampling enabled raises
    with pytest.raises(ValueError, match=msg):
        pred.skb.train_test_split(keep_subsampling=True)

    # same for cross-validation
    pred.skb.cross_validate()
    with pytest.raises(ValueError, match=msg):
        pred.skb.cross_validate(keep_subsampling=True)

    # subsampling OK but mismatch for some other reason
    msg = re.escape(
        "Found input variables with inconsistent numbers of samples: X: 99, y: 100"
    )
    X = skrub.var("X").skb.subsample(n=1000).skb.mark_as_X()
    y = skrub.var("y").skb.subsample(n=1000).skb.mark_as_y()
    pred = X.skb.apply(LogisticRegression(), y=y)
    with pytest.raises(ValueError, match=msg):
        pred.skb.train_test_split({"X": X_a[:-1], "y": y_a}, keep_subsampling=True)


#
# Bad arguments passed to eval()
#


def test_missing_var():
    e = skrub.var("a", 0) + skrub.var("b", 1)
    # we must provide either bindings for all vars or none
    assert e.skb.eval() == 1
    assert e.skb.eval({}) == 1
    with pytest.raises(
        (KeyError, RuntimeError),
        match=(
            "(Evaluation of node <Var 'b'> failed|No value has been provided for 'b')"
        ),
    ):
        e.skb.eval({"a": 10})


def test_X_y_instead_of_environment():
    with pytest.raises(
        TypeError,
        match=r"The `environment` passed to `eval\(\)` should be None or a dictionary",
    ):
        skrub.X().skb.eval(0)
    with pytest.raises(
        TypeError, match="`environment` should be a dictionary of input values"
    ):
        skrub.X().skb.make_learner().fit_transform(0)
    with pytest.raises(TypeError):
        skrub.X().skb.eval(X=0)
    with pytest.raises(TypeError):
        skrub.X().skb.make_learner().fit_transform(X=0)


def test_data_op_or_choice_in_environment():
    X = skrub.X()
    with pytest.raises(
        TypeError,
        match="The `environment` dict contains a skrub DataOp: <Var 'X'>",
    ):
        # likely mistake: passing a DataOp instead of an actual value.
        X.skb.eval({"X": X})

    alpha = skrub.choose_from([1.0, 2.0], name="alpha")
    with pytest.raises(
        TypeError,
        match="The `environment` dict contains a skrub choice: choose_from",
    ):
        (X + alpha).skb.eval({"X": 0, "alpha": alpha})


#
# Misc errors
#


def test_attribute_errors():
    # general case
    with pytest.raises(
        Exception, match=r"(?s)Evaluation of '\.something' failed.*AttributeError"
    ):
        skrub.X(0).something
    # added suggestion when the name exists in the .skb namespace
    with pytest.raises(Exception, match=r"(?sm).*^Did you mean `\.skb\.apply`"):
        skrub.X(0).apply
    with pytest.raises(
        AttributeError, match=r"`.skb.applied_estimator` only exists on"
    ):
        skrub.X().skb.applied_estimator


def test_call_method_errors():
    # In this case we have a preview value, so the access to `mark_as_X` fails
    # right away, before we add the call (when creating a GetAttr node)
    a = skrub.var("a", 0)
    with pytest.raises(
        (AttributeError, RuntimeError),
        match=r"(?sm).*^Did you mean `\.skb\.mark_as_X`",
    ):
        a.mark_as_X()
    a = skrub.var("a").mark_as_X()

    # In this case we have no preview value so the error only happens when we
    # call `.skb.eval()` and provide a value. By then the GetAttr + Call have
    # been collapsed in the graph into a CallMethod node, so the error
    # originates from a different place but it should provide the same info.
    with pytest.raises((AttributeError, RuntimeError)) as exc:
        a.skb.eval({"a": 0})
    assert "Did you mean `.skb.mark_as_X` and forget the `.skb`?" in "\n".join(
        traceback.format_exception(exc.value, exc.value, exc.tb)
    )


def test_concat_horizontal_numpy():
    a = skrub.var("a", skrub.datasets.toy_orders().orders)
    b = skrub.var("b", np.eye(3))
    with pytest.raises(Exception, match=".*can only be used with dataframes"):
        b.skb.concat([a], axis=1)
    with pytest.raises(Exception, match=".*should be passed a list of dataframes"):
        a.skb.concat([b], axis=1)


def test_concat_vertical_numpy():
    a = skrub.var("a", skrub.datasets.toy_orders().orders)
    b = skrub.var("b", np.eye(3))
    with pytest.raises(Exception, match=".*can only be used with dataframes"):
        b.skb.concat([a], axis=0)
    with pytest.raises(Exception, match=".*should be passed a list of dataframes"):
        a.skb.concat([b], axis=0)


def test_concat_needs_wrapping_in_list():
    a = skrub.var("a", skrub.datasets.toy_orders().orders)
    with pytest.raises(Exception, match=".*should be passed a list of dataframes"):
        a.skb.concat(a, axis=1)
    with pytest.raises(Exception, match=".*should be passed a list of dataframes"):
        a.skb.concat(a, axis=0)


def test_concat_axis_undefined():
    a = skrub.var("a", skrub.datasets.toy_orders().orders)
    with pytest.raises(Exception, match=".*one of 0 or 1"):
        a.skb.concat([a], axis=2)
    with pytest.raises(Exception, match=".*one of 0 or 1"):
        a.skb.concat([a], axis="foo")


def test_apply_instead_of_skb_apply():
    a = skrub.var("a", skrub.datasets.toy_orders().orders)
    with pytest.raises(
        Exception, match=r"(?sm).*^Did you mean `\.skb\.apply\('passthrough'\)`"
    ):
        a.apply("passthrough")
    with pytest.raises(
        Exception, match=r"(?sm).*^Did you mean `\.skb\.apply\(PassThrough\(\)\)`"
    ):
        a.apply(PassThrough())
    with pytest.raises(Exception, match=r"Evaluation of '.apply\(\)' failed\."):
        a.apply(int)


def test_apply_instead_of_apply_func():
    with pytest.raises(
        Exception,
        match=(
            r"(?sm).*Got a function instead.*^Did you mean to use"
            r" `\.skb\.apply_func\(\)`"
        ),
    ):
        skrub.X(0).skb.apply(lambda x: x)

    class Func:
        def __call__(self, x):
            return x

    with pytest.raises(
        Exception,
        match=r"(?sm).*Got a callable object instead.*"
        r"^Did you mean to use `\.skb\.apply_func\(\)`",
    ):
        skrub.X(0).skb.apply(Func())


def test_apply_class_not_instance():
    with pytest.raises(Exception, match=r"Please provide an instance"):
        skrub.X(0).skb.apply(skrub.TableVectorizer)


def test_apply_bad_type():
    with pytest.raises(
        Exception, match=r".*should be `None`, the string 'passthrough' or"
    ):
        skrub.X(0).skb.apply(...)


def test_apply_bad_string():
    with pytest.raises(
        Exception, match=r".*should be `None`, the string 'passthrough' or"
    ):
        skrub.X(0).skb.apply("pass through")


def test_method_call_failure():
    with pytest.raises(
        Exception,
        match=r"(?s)Evaluation of '.upper\(\)' failed.*takes no arguments \(1 given\)",
    ):
        skrub.var("a", "hello").upper(0)


def test_bad_names():
    with pytest.raises(
        TypeError, match=r"The `name` of a `skrub.var\(\)` must be a string"
    ):
        skrub.var(None)
    with pytest.raises(
        TypeError, match=r"The `name` of a `skrub.var\(\)` must be a string"
    ):
        skrub.var("a").skb.set_name(None)
    with pytest.raises(
        TypeError, match="`name` must be a string, got object of type: <class 'int'>"
    ):
        # forgot to pass the name
        skrub.var(0)
    with pytest.raises(
        TypeError,
        match="`name` must be a string or None, got object of type: <class 'int'>",
    ):
        # less likely to happen but for the sake of completeness
        (skrub.var("a") + 2).skb.set_name(0)
    with pytest.raises(ValueError, match="names starting with '_skrub_'"):
        skrub.var("_skrub_X")


def test_pass_df_instead_of_data_op():
    df = skrub.datasets.toy_orders().orders
    with pytest.raises(TypeError, match="You passed an actual DataFrame"):
        skrub.var("a").merge(df, on="ID")
    # this one is raised by pandas so we do not control the type or error
    # message but it fails early and is understandable
    with pytest.raises(Exception, match="Can only merge Series or DataFrame objects"):
        df.merge(skrub.var("a"), on="ID")
    pl = pytest.importorskip("polars")
    df = pl.from_pandas(df)
    with pytest.raises(TypeError, match="You passed an actual DataFrame"):
        skrub.var("a").join(df, on="ID")
    # this one is raised by polars so we do not control the type or error
    # message but it fails early and is understandable
    with pytest.raises(TypeError, match="expected .* to be a '?DataFrame'?"):
        df.join(skrub.var("a"), on="ID")


def test_make_grid_search_with_continuous_ranges():
    with pytest.raises(
        ValueError, match="Cannot use grid search with continuous numeric ranges"
    ):
        skrub.X().skb.apply(
            LogisticRegression(**skrub.choose_float(0.01, 10.0, log=True, name="C")),
            y=skrub.y(),
        ).skb.make_grid_search()


def test_data_op_with_circular_ref():
    # DataOp are not allowed to contain circular references as it would
    # complicate the implementation and there is probably no use case. We want
    # to get an understandable error and not an infinite loop or memory error.
    e = {}
    e["a"] = [0, {"b": e}]
    with pytest.raises(ValueError, match="DataOps cannot contain circular references"):
        skrub.as_data_op(e).skb.eval()


@pytest.mark.parametrize(
    "attribute",
    [
        "__copy__",
        "__float__",
        "__int__",
        "__reversed__",
        "_ipython_display_",
        "_ipython_canary_method_should_not_exist_",
        "_repr_intrinsic_type_",
    ],
)
def test_bad_attr(attribute):
    with pytest.raises(AttributeError):
        getattr(skrub.X(), attribute)


def test_unhashable():
    with pytest.raises(TypeError, match="unhashable type"):
        {skrub.X()}
    with pytest.raises(TypeError, match="unhashable type"):
        {skrub.choose_bool(name="b")}
    with pytest.raises(TypeError, match="unhashable type"):
        {skrub.choose_bool(name="b").if_else(0, 1)}


def test_int_column_names():
    with pytest.warns(match="Some dataframe column names are not strings"):
        skrub.X(pd.DataFrame({0: [1, 2]})).skb.apply("passthrough")
