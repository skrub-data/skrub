import itertools

from sklearn.base import BaseEstimator, TransformerMixin, clone

from . import _dataframe as sbd
from . import _selectors
from ._join_utils import pick_column_names


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
        results = []
        for col_name in sbd.column_names(X):
            results.append(
                _fit_transform_column(
                    sbd.col(X, col_name), self._columns, self.transformer
                )
            )
        return self._process_fit_transform_results(results, sbd.column_names(X))

    def _process_fit_transform_results(self, results, all_input_names):
        self.transformers_ = {}
        self.input_to_outputs_ = {}
        transformed_columns = []
        forbidden_names = set(all_input_names)
        for input_name, output_cols, transformer in results:
            if transformer is not None:
                suggested_names = _column_names(output_cols)
                output_names = pick_column_names(
                    suggested_names, forbidden_names - {input_name}
                )
                output_cols = _rename_columns(output_cols, output_names)
                forbidden_names.update(output_names)
                self.transformers_[input_name] = transformer
                self.input_to_outputs_[input_name] = output_names
            transformed_columns.extend(output_cols)

        self._output_names = _column_names(transformed_columns)
        self.used_inputs_ = list(self.transformers_.keys())
        self.produced_outputs_ = list(itertools.chain(*self.input_to_outputs_.values()))
        return sbd.dataframe_from_columns(*transformed_columns)

    def transform(self, X, y=None):
        del y
        transformed_columns = []
        for col_name in sbd.column_names(X):
            column = sbd.col(X, col_name)
            transformed_columns.extend(
                _transform_column(column, self.transformers_.get(col_name))
            )
        transformed_columns = _rename_columns(transformed_columns, self._output_names)
        return sbd.dataframe_from_columns(*transformed_columns)


def _prepare_transformer_input(transformer, column):
    # TODO better name
    if hasattr(transformer, "__univariate_transformer__"):
        return column
    return sbd.dataframe_from_columns(column)


def _fit_transform_column(column, columns_to_handle, transformer):
    col_name = sbd.name(column)
    if col_name not in columns_to_handle:
        return col_name, [column], None
    transformer = clone(transformer)
    if hasattr(transformer, "set_output"):
        df_module_name = sbd.dataframe_module_name(column)
        transformer.set_output(transform=df_module_name)
    transformer_input = _prepare_transformer_input(transformer, column)
    output = transformer.fit_transform(transformer_input)
    if output is NotImplemented:
        return col_name, [column], None
    output_cols = sbd.to_column_list(output)
    return col_name, output_cols, transformer


def _transform_column(column, transformer):
    if transformer is None:
        return [column]
    transformer_input = _prepare_transformer_input(transformer, column)
    output = transformer.transform(transformer_input)
    return sbd.to_column_list(output)


def _column_names(column_list):
    return [sbd.name(column) for column in column_list]


def _rename_columns(columns_list, new_names):
    return [sbd.rename(column, name) for (column, name) in zip(columns_list, new_names)]
