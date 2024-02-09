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

    created_outputs_ : list of str
        The names of columns in the output dataframe that were created by one
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
    >>> from sklearn.preprocessing import StandardScaler
    >>> df = pd.DataFrame(dict(A=[-10., 10.], B=[-10., 0.], C=[0., 10.]))
    >>> df
          A     B     C
    0 -10.0 -10.0   0.0
    1  10.0   0.0  10.0

    Fit a StandardScaler to each column in df:

    >>> scaling = Map(StandardScaler())
    >>> scaling.fit_transform(df)
         A    B    C
    0 -1.0 -1.0 -1.0
    1  1.0  1.0  1.0
    >>> scaling.transformers_
    {'A': StandardScaler(), 'B': StandardScaler(), 'C': StandardScaler()}

    We can restrict the columns on which the transformation is applied:

    >>> scaling = Map(StandardScaler(), cols=["A", "B"])
    >>> scaling.fit_transform(df)
         A    B     C
    0 -1.0 -1.0   0.0
    1  1.0  1.0  10.0

    We see that the scaling has not been applied to "C", which also does not
    appear in the transformers_:

    >>> scaling.transformers_
    {'A': StandardScaler(), 'B': StandardScaler()}
    >>> scaling.used_inputs_
    ['A', 'B']

    The transformer can return NotImplemented to indicate it cannot handle a
    given column.

    >>> from skrub._to_datetime import ToDatetime
    >>> df = pd.DataFrame(dict(birthday=["29/01/2024"], city=["London"]))
    >>> df
         birthday    city
    0  29/01/2024  London
    >>> df.dtypes
    birthday    object
    city        object
    dtype: object
    >>> ToDatetime().fit_transform(df["birthday"])
    0   2024-01-29
    Name: birthday, dtype: datetime64[ns]
    >>> ToDatetime().fit_transform(df["city"])
    NotImplemented
    >>> to_datetime = Map(ToDatetime())
    >>> transformed = to_datetime.fit_transform(df)
    >>> transformed
        birthday    city
    0 2024-01-29  London
    >>> transformed.dtypes
    birthday    datetime64[ns]
    city                object
    dtype: object
    >>> to_datetime.transformers_
    {'birthday': ToDatetime()}
    """

    def __init__(self, transformer, cols=_selectors.all(), n_jobs=None):
        self.transformer = transformer
        self.cols = cols
        self.n_jobs = n_jobs

    def __repr__(self) -> str:
        t_cls = self.transformer.__class__.__name__
        return f"<Transformer: {t_cls}.transform(col) for col in X[{self.cols}]>"

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
        self.output_to_input_ = {}
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
                self.output_to_input_.update(**{o: input_name for o in output_names})
            transformed_columns.extend(output_cols)

        self.all_outputs_ = _column_names(transformed_columns)
        self.used_inputs_ = list(self.transformers_.keys())
        self.created_outputs_ = list(itertools.chain(*self.input_to_outputs_.values()))
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
