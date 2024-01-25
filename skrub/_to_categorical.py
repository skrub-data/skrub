from sklearn.base import BaseEstimator

from . import _dataframe as sb
from ._dataframe import asdfapi, asnative, dfapi_ns

_HIGH_CARD_THRESHOLD = 30


class ToCategoricalCol(BaseEstimator):
    def fit_transform(self, column):
        if sb.is_categorical(column):
            self.output_native_dtype_ = sb.native_dtype(column)
            return column
        if sb.is_numeric(column) or sb.is_temporal(column):
            raise NotImplementedError()
        categories = list(sb.unique(column))
        if _HIGH_CARD_THRESHOLD <= len(categories):
            raise NotImplementedError()
        self.unknown_category_ = "skrub_unknown_category"
        self.categories_ = categories + [self.unknown_category_]
        self.output_native_dtype_ = sb.make_categorical_dtype_for(
            column, self.categories_
        )
        return self.transform(column)

    def transform(self, column):
        column = asdfapi(column)
        dfapi_categories = dfapi_ns(column).column_from_sequence(self.categories_)
        mask = asnative(column.is_in(dfapi_categories) | column.is_null())
        column = sb.where(asnative(column), mask, [self.unknown_category_])
        return sb.native_cast(asnative(column), self.output_native_dtype_)
