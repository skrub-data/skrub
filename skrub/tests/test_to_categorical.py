import pytest

from skrub import _dataframe as sbd
from skrub._single_column_transformer import RejectColumn
from skrub._to_categorical import ToCategorical


def test_to_categorical_pass(df_module):
    s = df_module.make_column("c", ["a", "b", None])
    assert not sbd.is_categorical(s)
    out = ToCategorical().fit_transform(s)
    assert sbd.is_categorical(out)
    # categorial columns are accepted
    assert ToCategorical().fit_transform(out) is out
    assert ToCategorical().fit(out).transform(out) is out
    # default behaviour accepts string columns
    expected = sbd.to_categorical(s)
    df_module.assert_column_equal(ToCategorical().fit_transform(s), expected)
    df_module.assert_column_equal(ToCategorical().fit(s).transform(s), expected)
    # also accepts int columns if accept_int is True
    i = df_module.make_column("c", [1, 2, None])
    expected = sbd.to_categorical(i)
    df_module.assert_column_equal(
        ToCategorical(accept_int=True).fit_transform(i), expected
    )
    df_module.assert_column_equal(
        ToCategorical(accept_int=True).fit(i).transform(i), expected
    )
    # once accepted during fit, transform works on any column regardless
    # of dtype
    f = df_module.make_column("c", [1.1, 2.2, None])
    assert sbd.is_categorical(ToCategorical().fit(s).transform(f))


@pytest.mark.parametrize(
    "accept_int,values",
    [
        (False, [1.1, 2.2, None]),  # float rejected always
        (True, [1.1, 2.2, None]),  # float rejected always
        (False, [1, 2, None]),  # int rejected when accept_int=False
    ],
)
def test_to_categorical_reject(df_module, accept_int, values):
    # reject columns based on accept_int parameter
    col = df_module.make_column("c", values)
    with pytest.raises(RejectColumn, match=".*does not contain only strings*"):
        ToCategorical(accept_int=accept_int).fit_transform(col)
