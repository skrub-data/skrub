import numpy as np
import pytest

import skrub
from skrub._utils import PassThrough


def test_concat_horizontal_numpy():
    a = skrub.var("a", skrub.toy_orders().orders)
    b = skrub.var("b", np.eye(3))
    with pytest.raises(Exception, match=".*can only be used with dataframes"):
        b.skb.concat_horizontal([a])
    with pytest.raises(Exception, match=".*should be passed a list of dataframes"):
        a.skb.concat_horizontal([b])


def test_concat_horizontal_needs_wrapping_in_list():
    a = skrub.var("a", skrub.toy_orders().orders)
    with pytest.raises(Exception, match=".*should be passed a list of dataframes"):
        a.skb.concat_horizontal(a)


def test_apply_instead_of_skb_apply():
    a = skrub.var("a", skrub.toy_orders().orders)
    with pytest.raises(Exception, match=r".*Did you mean `\.skb\.apply\(\)`"):
        a.apply("passthrough")
    with pytest.raises(Exception, match=r".*Did you mean `\.skb\.apply\(\)`"):
        a.apply(PassThrough())


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


def test_pass_df_instead_of_expr():
    df = skrub.toy_orders().orders
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
    with pytest.raises(TypeError, match="expected .* to be a DataFrame"):
        df.join(skrub.var("a"), on="ID")


def test_get_search_without_learner():
    with pytest.raises(
        AttributeError,
        match=(
            "`.skb.get_grid_search` only exists when the last step is a scikit-learn"
            " estimator"
        ),
    ):
        skrub.X().skb.get_grid_search()
