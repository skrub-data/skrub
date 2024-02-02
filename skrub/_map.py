import itertools

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone

from . import _dataframe as sbd
from . import _selectors
from ._join_utils import pick_column_names


class Map(TransformerMixin, BaseEstimator, auto_wrap_output_keys=()):
    """Map a transformer to columns in a dataframe.

    A separate clone of the transformer is applied to each column separately.
    Moreover, If the transformers' ``fit_transform`` returns ``NotImplemented``
    for a particular column, that column is passed through unchanged.

    Parameters
    ----------
    transformer : scikit-learn Transformer
        The transformer to map to the input dataframe's columns. For each
        column in the input dataframe, a clone of the transformer is created
        then ``fit_transform`` is called on a single-column dataframe. If the
        transformer has a ``__single_column_transformer__`` attribute,
        ``fit_transform`` is passed directly the column (a pandas or polars
        Series) rather than a DataFrame. ``fit_transform`` must return either a
        DataFrame, a Series, or a list of Series. ``fit_transform`` can return
        ``NotImplemented`` to indicate that this transformer does not apply to
        this column -- for example the ``ToDatetime`` transformer will return
        ``NotImplemented`` for numerical columns. In this case, the column will
        appear unchanged in the output.

    cols : str, sequence of str, or skrub selector, optional
        The columns to attempt to transform. Columns outside of this selection
        will be passed through unchanged, without attempting to call
        ``fit_transform`` on them. The default is to attempt transforming all
        columns.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a joblib ``parallel_backend`` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    all_inputs_ : list of str
        All column names in the input dataframe.

    used_inputs_ : list of str
        The names of columns that were transformed.

    all_outputs_ : list of str
        All column names in the output dataframe.

    produced_outputs_ : list of str
        The names of columns in the output dataframe that were produced by one
        of the fitted transformers.

    input_to_outputs_ : dict
        Maps the name of each column that was transformed to the list of the
        resulting columns' names in the output.

    transformers_ : dict
        Maps the name of each column that was transformed to the corresponding
        fitted transformer.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub._map import Map
    >>> from skrub._to_datetime import ToDatetime
    >>> df = pd.DataFrame(
    ...     dict(
    ...         A=[0.0, 1.0],
    ...         B=["02/02/2020", "22/03/2020"],
    ...         C=["02/02/2021", "22/03/2021"],
    ...     )
    ... )
    >>> Map(ToDatetime()).fit_transform(df)
         A          B          C
    0  0.0 2020-02-02 2021-02-02
    1  1.0 2020-03-22 2021-03-22
    >>> Map(ToDatetime(), cols=["A", "B"]).fit_transform(df)
         A          B           C
    0  0.0 2020-02-02  02/02/2021
    1  1.0 2020-03-22  22/03/2021
    """

    def __init__(self, transformer, cols=_selectors.all(), n_jobs=None):
        self.transformer = transformer
        self.cols = cols
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        del y
        self._columns = _selectors.make_selector(self.cols).select(X)
        results = []
        all_columns = sbd.column_names(X)
        parallel = Parallel(n_jobs=self.n_jobs)
        func = delayed(_fit_transform_column)
        results = parallel(
            func(sbd.col(X, col_name), self._columns, self.transformer)
            for col_name in all_columns
        )
        return self._process_fit_transform_results(results, X)

    def _process_fit_transform_results(self, results, X):
        all_input_names = sbd.column_names(X)
        self.all_inputs_ = all_input_names
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

        self.all_outputs_ = _column_names(transformed_columns)
        self.used_inputs_ = list(self.transformers_.keys())
        self.produced_outputs_ = list(itertools.chain(*self.input_to_outputs_.values()))
        return sbd.dataframe_like(X, *transformed_columns)

    def transform(self, X, y=None):
        del y
        transformed_columns = []
        for col_name in sbd.column_names(X):
            column = sbd.col(X, col_name)
            transformed_columns.extend(
                _transform_column(column, self.transformers_.get(col_name))
            )
        transformed_columns = _rename_columns(transformed_columns, self.all_outputs_)
        return sbd.dataframe_like(X, *transformed_columns)

    def get_feature_names_out(self):
        return self.all_outputs_


def _prepare_transformer_input(transformer, column):
    if hasattr(transformer, "__single_column_transformer__"):
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
