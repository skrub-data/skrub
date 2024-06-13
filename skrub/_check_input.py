import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from . import _join_utils, _utils
from ._dispatch import dispatch

__all__ = ["CheckInputDataFrame"]


def _column_names_to_strings(column_names):
    non_string = [c for c in column_names if not isinstance(c, str)]
    if not non_string:
        return column_names
    warnings.warn(
        f"Some column names are not strings: {non_string}. All column names"
        " must be strings; converting to strings."
    )
    return list(map(str, column_names))


def _deduplicated_column_names(column_names):
    duplicates = _utils.get_duplicates(column_names)
    if not duplicates:
        return column_names
    warnings.warn(
        f"Found duplicated column names: {duplicates}. Please make sure column names"
        " are unique. Renaming columns that have duplicated names."
    )
    return _join_utils.pick_column_names(column_names)


def _cleaned_column_names(column_names):
    return _deduplicated_column_names(_column_names_to_strings(column_names))


@dispatch
def _check_not_pandas_sparse(df):
    pass


@_check_not_pandas_sparse.specialize("pandas", argument_type="DataFrame")
def _check_not_pandas_sparse_pandas(df):
    import pandas as pd

    sparse_cols = [
        col for col in df.columns if isinstance(df[col].dtype, pd.SparseDtype)
    ]
    if sparse_cols:
        raise TypeError(
            f"Columns {sparse_cols} are sparse Pandas series, but dense "
            "data is required. Use ``df[col].sparse.to_dense()`` to convert "
            "a series from sparse to dense."
        )


def _check_is_dataframe(df):
    if not sbd.is_dataframe(df):
        raise TypeError(
            "Only pandas and polars DataFrames are supported. Cannot handle X of"
            f" type: {type(df)}."
        )


def _collect_lazyframe(df):
    if not sbd.is_lazyframe(df):
        return df
    warnings.warn(
        "At the moment, skrub only works on eager DataFrames, calling collect()."
    )
    return sbd.collect(df)


class CheckInputDataFrame(TransformerMixin, BaseEstimator):
    """Check the dataframe entering a skrub pipeline.

    This transformer ensures that:

    - The input is a dataframe.
        - Numpy arrays are converted to pandas dataframes with a warning.
    - The dataframe library is the same during ``fit`` and ``transform``, e.g.
      fitting on a polars dataframe and then transforming a pandas dataframe is
      not allowed.
        - A TypeError is raised otherwise.
    - Column names are unique strings.
        - Non-strings are cast to strings.
        - A random suffix is added to duplicated names.
        - If either of these operations is needed, a warning is emitted.
        - Only applies to pandas; polars column names are always unique strings.
    - The input is not sparse.
        - A TypeError is raised otherwise.
    - The input is not a ``LazyFrame``.
        - A ``LazyFrame`` is ``collect``ed with a warning.
    - The column names are the same during ``fit`` and ``transform``.
        - A ValueError is raised otherwise.

    Attributes
    ----------
    module_name_ : str
        The name of the dataframe module, 'polars' or 'pandas'.
    feature_names_in_ : list
        The column names of the input (before cleaning).
    n_features_in_ : int
        The number of input columns.
    feature_names_out_ : list of str
        The column names after converting to string and deduplication.
    """

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        del y
        X = self._handle_array(X)
        _check_is_dataframe(X)
        self.module_name_ = sbd.dataframe_module_name(X)
        # TODO check schema (including dtypes) not just names.
        # Need to decide how strict we should be about types
        column_names = sbd.column_names(X)
        self.feature_names_in_ = column_names
        self.n_features_in_ = len(column_names)
        self.feature_names_out_ = _cleaned_column_names(column_names)
        if sbd.column_names(X) != self.feature_names_out_:
            X = sbd.set_column_names(X, self.feature_names_out_)
        _check_not_pandas_sparse(X)
        X = _collect_lazyframe(X)
        return X

    def transform(self, X):
        check_is_fitted(self, "module_name_")
        X = self._handle_array(X)
        _check_is_dataframe(X)
        module_name = sbd.dataframe_module_name(X)
        if module_name != self.module_name_:
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
        if sbd.column_names(X) != self.feature_names_out_:
            X = sbd.set_column_names(X, self.feature_names_out_)
        _check_not_pandas_sparse(X)
        X = _collect_lazyframe(X)
        return X

    def _handle_array(self, X):
        if not isinstance(X, np.ndarray):
            return X
        if X.ndim != 2:
            raise ValueError(
                "Input should be a DataFrame. Found an array with incompatible shape:"
                f" {X.shape}."
            )
        warnings.warn(
            "Only pandas and polars DataFrames are supported, but input is a Numpy"
            " array. Please convert Numpy arrays to DataFrames before passing them to"
            " skrub transformers. Converting to pandas DataFrame with columns"
            " ['0', '1', …]."
        )
        import pandas as pd

        columns = list(map(str, range(X.shape[1])))
        X = pd.DataFrame(X, columns=columns)
        return X

    # set_output api compatibility

    def get_feature_names_out(self):
        return self.feature_names_out_
