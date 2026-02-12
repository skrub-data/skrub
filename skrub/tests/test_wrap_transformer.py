from sklearn.preprocessing import OrdinalEncoder

from skrub import ApplyToCols
from skrub import selectors as s
from skrub._apply_to_frame import ApplyToFrame
from skrub._to_datetime import ToDatetime
from skrub._wrap_transformer import wrap_transformer


def test_wrap_transformer():
    t = wrap_transformer(ToDatetime(), s.all())
    assert isinstance(t, ApplyToCols)
    t = wrap_transformer(OrdinalEncoder(), s.all())
    assert isinstance(t, ApplyToFrame)
    t = wrap_transformer(OrdinalEncoder(), s.all(), columnwise=True)
    assert isinstance(t, ApplyToCols)
