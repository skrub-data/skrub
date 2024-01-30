import warnings

from sklearn.base import BaseEstimator, TransformerMixin

from . import _dataframe as sbd
from . import _utils

# auto_wrap_output_keys = () is so that the TransformerMixin does not wrap
# transform or provide set output (we always produce dataframes of the correct
# type with the correct columns and we don't want the wrapper.) other ways to
# disable it would be not inheriting from TransformerMixin, not defining
# get_feature_names_out


class CheckInputDataFrame(TransformerMixin, BaseEstimator, auto_wrap_output_keys=()):
    def fit(self, X, y=None):
        del y
        module_name = sbd.dataframe_module_name(X)
        if module_name is None:
            raise TypeError(
                "Only pandas and polars DataFrames are"
                f" supported. Cannot handle X of type: {type(X)}"
            )
        self.module_name_ = module_name
        # TODO check schema (including dtypes) not just names.
        # Need to decide how strict we should be about types
        column_names = sbd.column_names(X)
        _utils.check_duplicated_column_names(column_names)
        self.feature_names_in_ = column_names
        return self

    def transform(self, X):
        module_name = sbd.dataframe_module_name(X)
        if module_name is None:
            raise TypeError(
                "Only pandas DataFrames and polars DataFrames and LazyFrames are"
                f" supported. Cannot handle X of type: {type(X)}"
            )
        if module_name != self.module_name_:
            # TODO should this be a warning instead?
            raise TypeError(
                f"Pipeline was fitted to a {self.module_name_} dataframe "
                f"but is being applied to a {module_name} dataframe. "
                "This is likely to produce errors and is not supported."
            )
        column_names = sbd.column_names(X)
        if column_names != self.feature_names_in_:
            import difflib

            diff = "\n".join(
                difflib.Differ().compare(self.feature_names_in_, column_names)
            )
            message = (
                f"Columns of dataframes passed to fit() and transform() differ:\n{diff}"
            )
            raise ValueError(message)
        if sbd.is_lazyframe(X):
            warnings.warn(
                "At the moment, skrub only works on eager DataFrames, calling collect()"
            )
            X = sbd.collect(X)
        return X

    def get_feature_names_out(self):
        return self.feature_names_in_
