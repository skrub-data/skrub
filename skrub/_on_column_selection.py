from sklearn.base import BaseEstimator, TransformerMixin, clone

from . import _dataframe as sbd
from . import _selectors
from ._join_utils import pick_column_names


class OnColumnSelection(TransformerMixin, BaseEstimator, auto_wrap_output_keys=()):
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

    created_outputs_ : list of str
        The names of columns in the output dataframe that were created by the
        fitted transformer.

    transformer_ : Transformer
        The fitted transformer.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> df = pd.DataFrame(np.eye(4) * np.logspace(0, 3, 4), columns=list("abcd"))
    >>> df
         a     b      c       d
    0  1.0   0.0    0.0     0.0
    1  0.0  10.0    0.0     0.0
    2  0.0   0.0  100.0     0.0
    3  0.0   0.0    0.0  1000.0
    >>> from sklearn.decomposition import PCA
    >>> from skrub._on_column_selection import OnColumnSelection
    >>> OnColumnSelection(PCA(n_components=2)).fit_transform(df).round(2)
         pca0   pca1
    0 -249.01 -33.18
    1 -249.04 -33.68
    2 -252.37  66.64
    3  750.42   0.22

    We can restrict the transformer to a subset of columns:

    >>> pca = OnColumnSelection(PCA(n_components=2), cols=["a", "b"])
    >>> pca.fit_transform(df).round(2)
           c       d  pca0  pca1
    0    0.0     0.0 -2.52  0.67
    1    0.0     0.0  7.50  0.00
    2  100.0     0.0 -2.49 -0.33
    3    0.0  1000.0 -2.49 -0.33
    >>> pca.used_inputs_
    ['a', 'b']
    >>> pca.created_outputs_
    ['pca0', 'pca1']
    >>> pca.transformer_
    PCA(n_components=2)
    """

    def __init__(self, transformer, cols=_selectors.all()):
        self.transformer = transformer
        self.cols = cols

    def __repr__(self) -> str:
        t_cls = self.transformer.__class__.__name__
        return f"<Transformer: {t_cls}.transform(X[{self.cols}])>"

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
        self.created_outputs_ = self._transformed_output_names
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
