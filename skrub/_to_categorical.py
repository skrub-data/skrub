from sklearn.base import BaseEstimator

from . import _dataframe as sbd
from . import _utils
from ._dataframe import asdfapi, asnative, dfapi_ns


class ToCategorical(BaseEstimator):
    __univariate_transformer__ = True

    def __init__(self, cardinality_threshold=40):
        self.cardinality_threshold = cardinality_threshold

    def fit_transform(self, column):
        if sbd.is_categorical(column):
            self.output_native_dtype_ = sbd.native_dtype(column)
            return column
        if not sbd.is_string(column):
            return NotImplemented
        categories = list(sbd.unique(column))
        if self.cardinality_threshold <= len(categories):
            return NotImplemented
        token = _utils.random_string()
        self.unknown_category_ = f"skrub_unknown_category_{token}"
        self._categories = categories + [self.unknown_category_]
        self.output_native_dtype_ = sbd.make_categorical_dtype_for(
            column, self._categories
        )
        return self.transform(column)

    def transform(self, column):
        if sbd.is_categorical(column):
            return column
        column = asdfapi(column)
        dfapi_categories = dfapi_ns(column).column_from_sequence(self._categories)
        mask = asnative(column.is_in(dfapi_categories) | column.is_null())
        column = sbd.where(asnative(column), mask, [self.unknown_category_])
        return sbd.native_cast(asnative(column), self.output_native_dtype_)
