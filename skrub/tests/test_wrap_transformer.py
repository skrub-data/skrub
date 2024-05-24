from sklearn.preprocessing import OrdinalEncoder

from skrub import _selectors as s
from skrub._on_each_column import OnEachColumn
from skrub._on_subframe import OnSubFrame
from skrub._to_datetime import ToDatetime
from skrub._wrap_transformer import wrap_transformer


def test_wrap_transformer():
    t = wrap_transformer(ToDatetime(), s.all())
    assert isinstance(t, OnEachColumn)
    t = wrap_transformer(OrdinalEncoder(), s.all())
    assert isinstance(t, OnSubFrame)
    t = wrap_transformer(OrdinalEncoder(), s.all(), columnwise=True)
    assert isinstance(t, OnEachColumn)
