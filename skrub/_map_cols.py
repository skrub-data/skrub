import warnings

from sklearn.base import BaseEstimator, TransformerMixin, clone

from . import _dataframe as sbd
from . import _selectors
from ._dataframe import asdfapi, asnative, dfapi_ns


class MapCols(TransformerMixin, BaseEstimator):
    def __init__(self, column_transformer, cols=_selectors.all()):
        self.column_transformer = column_transformer
        self.cols = cols

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        del y
        self._columns = _selectors.make_selector(self.cols).select(X)
        self.transformers_ = {}
        transformed_columns = []
        df_module_name = sbd.dataframe_module_name(X)
        ns = dfapi_ns(X)
        X = asdfapi(X)
        for col_name in X.column_names:
            column = X.col(col_name)
            if col_name in self._columns:
                transformer = clone(self.column_transformer)
                if hasattr(transformer, "set_output"):
                    transformer.set_output(transform=df_module_name)
                transformer_input = _prepare_transformer_input(transformer, column)
                try:
                    output = transformer.fit_transform(transformer_input)
                except NotImplementedError:
                    transformed_columns.append(column)
                else:
                    transformed_columns.extend(sbd.to_dfapi_column_list(output))
                    self.transformers_[col_name] = transformer
            else:
                transformed_columns.append(column)
        # TODO find a way to know if a column is already persisted and avoid warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed_columns = [c.persist() for c in transformed_columns]
        return asnative(ns.dataframe_from_columns(*transformed_columns))

    def transform(self, X, y=None):
        del y
        transformed_columns = []
        X = asdfapi(X)
        for col_name in X.column_names:
            column = X.col(col_name)
            if col_name in self.transformers_:
                transformer = self.transformers_[col_name]
                transformer_input = _prepare_transformer_input(transformer, column)
                output = transformer.transform(transformer_input)
                transformed_columns.extend(sbd.to_dfapi_column_list(output))
            else:
                transformed_columns.append(column)
        ns = dfapi_ns(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            transformed_columns = [c.persist() for c in transformed_columns]
        return asnative(ns.dataframe_from_columns(*transformed_columns))


def _prepare_transformer_input(transformer, dfapi_column):
    # TODO better name
    if hasattr(transformer, "__univariate_transformer__"):
        return asnative(dfapi_column)
    ns = dfapi_ns(asnative(dfapi_column))
    return asnative(ns.dataframe_from_columns(dfapi_column.persist()))
