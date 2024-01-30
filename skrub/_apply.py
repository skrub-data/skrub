from sklearn.base import BaseEstimator, TransformerMixin, clone

from . import _dataframe as sbd
from . import _selectors
from ._join_utils import pick_column_names

# auto_wrap_output_keys = () is so that the TransformerMixin does not wrap
# transform or provide set output (we always produce dataframes of the correct
# type with the correct columns and we don't want the wrapper.) other ways to
# disable it would be not inheriting from TransformerMixin, not defining
# get_feature_names_out


class Apply(TransformerMixin, BaseEstimator, auto_wrap_output_keys=()):
    def __init__(self, transformer, cols=_selectors.all()):
        self.transformer = transformer
        self.cols = cols

    def fit_transform(self, X, y=None):
        self.all_inputs_ = sbd.column_names(X)
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
        self.all_outputs_ = passthrough_names + self._transformed_output_names
        self.feature_names_in_ = self.all_inputs_
        return sbd.concat_horizontal(passthrough, transformed)

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        to_transform = _selectors.select(X, self._columns)
        passthrough = _selectors.select(X, _selectors.inv(self._columns))
        transformed = self.transformer_.transform(to_transform)
        transformed = sbd.set_column_names(transformed, self._transformed_output_names)
        return sbd.concat_horizontal(passthrough, transformed)

    def get_feature_names_out(self):
        return self.all_outputs_
