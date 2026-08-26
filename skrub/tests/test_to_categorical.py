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
    # default behaviour accepts integer and string
    # columns, but not float
    i = df_module.make_column("c", [1, 2, None])
    expected = sbd.to_categorical(i)
    df_module.assert_column_equal(
        ToCategorical(accept_numeric="int").fit_transform(i), expected
    )
    df_module.assert_column_equal(
        ToCategorical(accept_numeric="int").fit(i).transform(i), expected
    )
    # unless accept_numeric is "float", in which case floats are accepted
    f = df_module.make_column("c", [1.1, 2.2, None])
    expected = sbd.to_categorical(f)
    df_module.assert_column_equal(
        ToCategorical(accept_numeric="all").fit_transform(f), expected
    )
    df_module.assert_column_equal(
        ToCategorical(accept_numeric="all").fit(f).transform(f), expected
    )
    # but once accepted during fit, transform works on any column regardless
    # of dtype
    assert sbd.is_categorical(ToCategorical().fit(s).transform(f))


@pytest.mark.parametrize(
    "accept_numeric,values",
    [
        ("int", [1.1, 2.2, None]),  # float rejected when accept_numeric="int"
        (None, [1, 2, None]),  # int rejected when accept_numeric=None
        (None, [1.1, 2.2, None]),  # float rejected when accept_numeric=None
    ],
)
def test_to_categorical_reject(df_module, accept_numeric, values):
    # reject columns based on accept_numeric parameter
    col = df_module.make_column("c", values)
    with pytest.raises(RejectColumn, match=".*does not contain strings or*"):
        ToCategorical(accept_numeric=accept_numeric).fit_transform(col)
