from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from skrub import ApplyOnEachCol
from skrub import selectors as s
from skrub._apply_to_frame import ApplySubFrame
from skrub._datetime_encoder import DatetimeEncoder
from skrub._to_datetime import ToDatetime
from skrub._wrap_transformer import wrap_transformer


def test_wrap_transformer():
    t = wrap_transformer(ToDatetime(), s.all())
    assert isinstance(t, ApplyOnEachCol)
    t = wrap_transformer(OrdinalEncoder(), s.all())
    assert isinstance(t, ApplySubFrame)
    t = wrap_transformer(OrdinalEncoder(), s.all(), columnwise=True)
    assert isinstance(t, ApplyOnEachCol)
    t = wrap_transformer(make_pipeline(DatetimeEncoder(), StandardScaler()), s.all())
    assert isinstance(t, ApplyOnEachCol)
    t = wrap_transformer(make_pipeline(StandardScaler()), s.all())
    assert isinstance(t, ApplySubFrame)
