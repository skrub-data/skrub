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
        if sbd.is_numeric(column) or sbd.is_anydate(column):
            raise NotImplementedError()
        categories = list(sbd.unique(column))
        if self.cardinality_threshold <= len(categories):
            raise NotImplementedError()
        token = _utils.random_string()
        self.unknown_category_ = f"skrub_unknown_category_{token}"
        self.categories_ = categories + [self.unknown_category_]
        self.output_native_dtype_ = sbd.make_categorical_dtype_for(
            column, self.categories_
        )
        return self.transform(column)

    def transform(self, column):
        column = asdfapi(column)
        dfapi_categories = dfapi_ns(column).column_from_sequence(self.categories_)
        mask = asnative(column.is_in(dfapi_categories) | column.is_null())
        column = sbd.where(asnative(column), mask, [self.unknown_category_])
        return sbd.native_cast(asnative(column), self.output_native_dtype_)
