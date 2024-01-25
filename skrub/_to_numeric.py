from sklearn.base import BaseEstimator

from ._dataframe import asdfapi, asnative, dfapi_ns, is_numeric


def _to_numeric(column):
    column = asdfapi(column)
    ns = dfapi_ns(column)
    if is_numeric(column):
        return asnative(column)
    try:
        column = column.cast(ns.Int64())
        return asnative(column)
    except Exception:
        pass
    try:
        column = column.cast(ns.Float64())
        return asnative(column)
    except Exception:
        pass
    raise ValueError(f"Could not convert to numeric dtype: {column}")


class ToNumericCol(BaseEstimator):
    def fit_transform(self, column):
        column = asdfapi(column)
        try:
            numeric = asdfapi(_to_numeric(column))
            self.output_dtype_ = numeric.dtype
            return asnative(numeric)
        except Exception:
            raise NotImplementedError()

    def transform(self, column):
        column = asdfapi(column)
        return asnative(column.cast(self.output_dtype_))
