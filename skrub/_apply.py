from sklearn.base import BaseEstimator, TransformerMixin, clone

from . import _dataframe as sbd
from . import _selectors
from ._join_utils import pick_column_names


class Apply(TransformerMixin, BaseEstimator):
    def __init__(self, transformer, cols=_selectors.all()):
        self.transformer = transformer
        self.cols = cols

    def fit_transform(self, X, y=None):
        self._columns = _selectors.make_selector(self.cols).select(X)
        to_transform = _selectors.select(X, self._columns)
        passthrough = _selectors.select(X, _selectors.inv(self._columns))
        self.transformer_ = clone(self.transformer)
        if hasattr(self.transformer_, "set_output"):
            df_module_name = sbd.dataframe_module_name(X)
            self.transformer_.set_output(transform=df_module_name)
        transformed = self.transformer_.fit_transform(to_transform, y)
        passthrough_names = sbd.column_names(passthrough)
        self._transformed_output_names = pick_column_names(
            sbd.column_names(transformed), forbidden_names=passthrough_names
        )
        transformed = sbd.set_column_names(transformed, self._transformed_output_names)
        self.used_inputs_ = self._columns
        self.produced_outputs_ = self._transformed_output_names
        self._output_names = passthrough_names + self._transformed_output_names
        return sbd.concat_horizontal(passthrough, transformed)

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        to_transform = _selectors.select(X, self._columns)
        passthrough = _selectors.select(X, _selectors.inv(self._columns))
        transformed = self.transformer_.transform(to_transform)
        transformed = sbd.set_column_names(transformed, self._transformed_output_names)
        return sbd.concat_horizontal(passthrough, transformed)
