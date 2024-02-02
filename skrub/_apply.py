from sklearn.base import BaseEstimator, TransformerMixin, clone

from . import _dataframe as sbd
from . import _selectors
from ._join_utils import pick_column_names


class Apply(TransformerMixin, BaseEstimator, auto_wrap_output_keys=()):
    """Apply a transformer to part of a dataframe.

    A subset of the dataframe is selected and passed to the transformer (as a
    single input). This is different from ``Map`` which fits a separate clone
    of the transformer to each selected column independently.

    Parameters
    ----------
    transformer : scikit-learn Transformer
        The transformer to apply to the selected columns. ``fit_transform`` and
        ``transform`` must return a DataFrame. The resulting dataframe will
        appear as the last columns of the output dataframe. Unselected columns
        will appear unchanged in the output.

    cols : str, sequence of str, or skrub selector, optional
        The columns to attempt to transform. Columns outside of this selection
        will be passed through unchanged, without calling ``fit_transform`` on
        them. The default is transform all columns.

    Attributes
    ----------
    all_inputs_ : list of str
        All column names in the input dataframe.

    used_inputs_ : list of str
        The names of columns that were transformed.

    all_outputs_ : list of str
        All column names in the output dataframe.

    produced_outputs_ : list of str
        The names of columns in the output dataframe that were produced the
        fitted transformer.

    transformer_ : Transformer
        The fitted transformer.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(dict(A=[-10.0, 10.0], B=[-10.0, 10.0], C=[-10.0, 10.0]))
    >>> df
          A     B     C
    0 -10.0 -10.0 -10.0
    1  10.0  10.0  10.0
    >>> from skrub._apply import Apply
    >>> from sklearn.preprocessing import StandardScaler
    >>> Apply(StandardScaler()).fit_transform(df)
         A    B    C
    0 -1.0 -1.0 -1.0
    1  1.0  1.0  1.0
    >>> Apply(StandardScaler(), ["A", "B"]).fit_transform(df)
          C    A    B
    0 -10.0 -1.0 -1.0
    1  10.0  1.0  1.0
    """

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
