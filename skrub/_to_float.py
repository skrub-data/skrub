from sklearn.base import BaseEstimator

from . import _dataframe as sbd


@sbd.dispatch
def _to_float32(col):
    raise NotImplementedError()


@_to_float32.specialize("pandas")
def _to_float32_pandas(col):
    return col.astype("float32")


@_to_float32.specialize("polars")
def _to_float32_polars(col):
    import polars as pl

    return col.astype(pl.Float32)


class ToFloat32(BaseEstimator):
    __univariate_transformer__ = True

    def fit_transform(self, column):
        if not sbd.is_numeric(column):
            return NotImplemented
        try:
            numeric = sbd.to_numeric(column)
            self.output_native_dtype_ = sbd.native_dtype(numeric)
            return numeric
        except Exception:
            return NotImplemented

    def transform(self, column):
        return sbd.to_numeric(column, dtype=self.output_native_dtype_)
