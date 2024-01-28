from sklearn.base import BaseEstimator, TransformerMixin, clone

from . import _dataframe as sbd
from . import _selectors


class Map(TransformerMixin, BaseEstimator):
    def __init__(self, transformer, cols=_selectors.all()):
        self.transformer = transformer
        self.cols = cols

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        del y
        self._columns = _selectors.make_selector(self.cols).select(X)
        self.transformers_ = {}
        transformed_columns = []
        self.used_inputs_ = []
        self.input_to_outputs_ = {}
        self.produced_outputs_ = []
        df_module_name = sbd.dataframe_module_name(X)
        for col_name in sbd.column_names(X):
            column = sbd.col(X, col_name)
            if col_name in self._columns:
                transformer = clone(self.transformer)
                if hasattr(transformer, "set_output"):
                    transformer.set_output(transform=df_module_name)
                transformer_input = _prepare_transformer_input(transformer, column)
                output = transformer.fit_transform(transformer_input)
                if output is NotImplemented:
                    transformed_columns.append(column)
                else:
                    output_cols = sbd.to_column_list(output)
                    output_col_names = [sbd.name(c) for c in output_cols]
                    transformed_columns.extend(output_cols)
                    self.transformers_[col_name] = transformer
                    self.used_inputs_.append(col_name)
                    self.input_to_outputs_[col_name] = output_col_names
                    self.produced_outputs_.extend(output_col_names)
            else:
                transformed_columns.append(column)
        return sbd.dataframe_from_columns(*transformed_columns)

    def transform(self, X, y=None):
        del y
        transformed_columns = []
        for col_name in sbd.column_names(X):
            column = sbd.col(X, col_name)
            if col_name in self.transformers_:
                transformer = self.transformers_[col_name]
                transformer_input = _prepare_transformer_input(transformer, column)
                output = transformer.transform(transformer_input)
                transformed_columns.extend(sbd.to_column_list(output))
            else:
                transformed_columns.append(column)
        return sbd.dataframe_from_columns(*transformed_columns)


def _prepare_transformer_input(transformer, column):
    # TODO better name
    if hasattr(transformer, "__univariate_transformer__"):
        return column
    return sbd.dataframe_from_columns(column)
