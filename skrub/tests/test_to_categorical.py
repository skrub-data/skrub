import pytest

from skrub import _dataframe as sbd
from skrub._on_each_column import RejectColumn
from skrub._to_categorical import ToCategorical


def test_to_categorical(df_module):
    s = df_module.make_column("c", ["a", "b", None])
    assert not sbd.is_categorical(s)
    out = ToCategorical().fit_transform(s)
    assert sbd.is_categorical(out)
    # categorial columns are accepted
    assert ToCategorical().fit_transform(out) is out
    assert ToCategorical().fit(out).transform(out) is out
    # non-string, non-categorical columns are rejected
    f = df_module.make_column("c", [1.1, 2.2, None])
    with pytest.raises(RejectColumn, match=".*does not contain strings"):
        ToCategorical().fit(f)
    # but once accepted during fit, transform works on any column
    assert sbd.is_categorical(ToCategorical().fit(s).transform(f))
