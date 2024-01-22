from sklearn.base import BaseEstimator

from skrub._dataframe import (
    asdfapi,
    asnative,
    dfns,
    is_categorical,
    is_datetime,
    is_numeric,
    native_cast,
    native_dtype,
    skrub_namespace,
    unique,
    where,
)

_HIGH_CARD_THRESHOLD = 30


class ToCategoricalCol(BaseEstimator):
    def fit_transform(self, column):
        if is_categorical(column):
            self.output_native_dtype_ = native_dtype(column)
            return column
        if is_numeric(column) or is_datetime(column):
            raise NotImplementedError()
        categories = list(unique(column))
        if _HIGH_CARD_THRESHOLD <= len(categories):
            raise NotImplementedError()
        skrub_ns = skrub_namespace(column)
        self.unknown_category_ = "skrub_unknown_category"
        self.categories_ = categories + [self.unknown_category_]
        self.output_native_dtype_ = skrub_ns.make_categorical_dtype(self.categories_)
        return self.transform(column)

    def transform(self, column):
        column = asdfapi(column)
        dfapi_categories = dfns(column).column_from_sequence(self.categories_)
        mask = asnative(column.is_in(dfapi_categories) | column.is_null())
        column = where(asnative(column), mask, [self.unknown_category_])
        return native_cast(asnative(column), self.output_native_dtype_)
