from sklearn.base import BaseEstimator

from . import _dataframe as sbd
from ._dispatch import dispatch
from ._exceptions import RejectColumn


@dispatch
def _to_float32(col):
    raise NotImplementedError()


@_to_float32.specialize("pandas")
def _to_float32_pandas(col):
    return col.astype("float32")


@_to_float32.specialize("polars")
def _to_float32_polars(col):
    import polars as pl

    return col.cast(pl.Float32)


class ToFloat32(BaseEstimator):
    __single_column_transformer__ = True

    def fit_transform(self, column):
        if not sbd.is_numeric(column):
            raise RejectColumn(
                f"Column {sbd.name(column)!r} does not have a numeric dtype."
            )
        return self.transform(column)

    def transform(self, column):
        return _to_float32(column)

    def fit(self, column):
        self.fit_transform(column)
        return self
