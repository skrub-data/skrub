import functools
import itertools
import re
import textwrap

from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from . import _selectors, _utils
from ._join_utils import pick_column_names

__all__ = ["OnEachColumn", "SingleColumnTransformer", "RejectColumn"]

_SINGLE_COL_LINE = (
    "``{class_name}`` is a type of single-column transformer. Unlike most scikit-learn"
    " estimators, its ``fit``, ``transform`` and ``fit_transform`` methods expect a"
    " single column (a pandas or polars Series) rather than a full dataframe. To apply"
    " this transformer to one or more columns in a dataframe, use it as a parameter in"
    " a ``skrub.TableVectorizer`` or ``sklearn.compose.ColumnTransformer``. In the"
    " ``ColumnTransformer``, pass a single column:"
    " ``make_column_transformer(({class_name}(), 'col_name_1'), ({class_name}(),"
    " 'col_name_2'))`` instead of ``make_column_transformer(({class_name}(),"
    " ['col_name_1', 'col_name_2']))``."
)
_SINGLE_COL_PARAGRAPH = textwrap.fill(
    _SINGLE_COL_LINE, initial_indent="    ", subsequent_indent="    "
)
_SINGLE_COL_NOTE = f".. note::\n\n{_SINGLE_COL_PARAGRAPH}\n"


class RejectColumn(ValueError):
    """Used by single-column transformers to indicate they do not apply to a column.

    >>> import pandas as pd
    >>> from skrub._to_datetime import ToDatetime
    >>> df = pd.DataFrame(dict(a=['2020-02-02'], b=[12.5]))
    >>> ToDatetime().fit_transform(df['a'])
    0   2020-02-02
    Name: a, dtype: datetime64[ns]
    >>> ToDatetime().fit_transform(df['b'])
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Column 'b' does not contain strings.
    """

    pass


class SingleColumnTransformer(BaseEstimator):
    """Base class for single-column transformers.

    Such transformers are applied independently to each column by
    ``OnEachColumn``; see the docstring of ``OnEachColumn`` for more
    information.

    Single-column transformers are not required to inherit from this class in
    order to work with ``OnEachColumn``, however doing so avoids some
    boilerplate:

        - The required ``__single_column_transformer__`` attribute is set.
        - ``fit`` is defined (calls ``fit_transform`` and discards the result).
        - ``fit``, ``transform`` and ``fit_transform`` are wrapped to check
          that the input is a single column and raise a ``ValueError`` with a
          helpful message when it is not.
        - A note about single-column transformers (vs dataframe transformers)
          is added after the summary line of the docstring.

    Subclasses must define ``fit_transform`` and ``transform`` (or inherit them
    from another superclass).
    """

    __single_column_transformer__ = True

    def fit(self, column, y=None):
        """Fit the transformer.

        Subclasses should implement ``fit_transform`` and ``transform``.

        Parameters
        ----------
        column : a pandas or polars Series
            Unlike most scikit-learn transformers, single-column transformers
            transform a single column, not a whole dataframe.

        y : column or dataframe
            Prediction targets.

        Returns
        -------
        self
            The fitted transformer.
        """
        self.fit_transform(column, y=y)
        return self

    def _check_single_column(self, column, function_name):
        class_name = self.__class__.__name__
        if sbd.is_dataframe(column):
            raise ValueError(
                f"``{class_name}.{function_name}`` should be passed a single column,"
                " not a dataframe. "
                + _SINGLE_COL_LINE.format(class_name=class_name)
            )
        if not sbd.is_column(column):
            raise ValueError(
                f"``{class_name}.{function_name}`` expects the first argument X "
                "to be a column (a pandas or polars Series). "
                f"Got X with type: {column.__class__.__name__}."
            )
        return column

    def __init_subclass__(subclass, **kwargs):
        super().__init_subclass__(**kwargs)
        if subclass.__doc__ is not None:
            subclass.__doc__ = _insert_after_first_paragraph(
                subclass.__doc__,
                _SINGLE_COL_NOTE.format(class_name=subclass.__name__),
            )
        for method in "fit", "fit_transform", "transform", "partial_fit":
            if method in subclass.__dict__:
                wrapped = _wrap_add_check_single_column(getattr(subclass, method))
                setattr(subclass, method, wrapped)


def _wrap_add_check_single_column(f):
    # as we have only a few predefined functions to handle, using their exact
    # name and signature in the wrapper definition gives better tracebacks and
    # autocompletion than just functools.wraps / setting __name__ and
    # __signature__
    if f.__name__ == "fit":

        @functools.wraps(f)
        def fit(self, X, y=None):
            self._check_single_column(X, f.__name__)
            return f(self, X, y=y)

        return fit
    elif f.__name__ == "partial_fit":

        @functools.wraps(f)
        def partial_fit(self, X, y=None):
            self._check_single_column(X, f.__name__)
            return f(self, X, y=y)

        return partial_fit

    elif f.__name__ == "fit_transform":

        @functools.wraps(f)
        def fit_transform(self, X, y=None):
            self._check_single_column(X, f.__name__)
            return f(self, X, y=y)

        return fit_transform
    else:
        assert f.__name__ == "transform", f.__name__

        @functools.wraps(f)
        def transform(self, X):
            self._check_single_column(X, f.__name__)
            return f(self, X)

        return transform


def _insert_after_first_paragraph(document, text_to_insert):
    split_doc = document.splitlines(True)
    indent = min(
        (
            len(m.group(1))
            for line in split_doc[1:]
            if (m := re.match(r"^( *)\S", line)) is not None
        ),
        default=0,
    )
    doc_lines = iter(split_doc)
    output_lines = []
    for line in doc_lines:
        output_lines.append(line)
        if line.strip():
            break
    for line in doc_lines:
        output_lines.append(line)
        if not line.strip():
            break
    else:
        output_lines.append("\n")
    for line in text_to_insert.splitlines(True):
        output_lines.append(line if not line.strip() else " " * indent + line)
    output_lines.append("\n")
    output_lines.extend(doc_lines)
    return "".join(output_lines)


class OnEachColumn(TransformerMixin, BaseEstimator):
    """Map a transformer to columns in a dataframe.

    A separate clone of the transformer is applied to each column separately.
    Moreover, if ``allow_reject`` is ``True`` and the transformers'
    ``fit_transform`` raises a ``RejectColumn`` exception for a particular
    column, that column is passed through unchanged. If ``allow_reject`` is
    ``False``, ``RejectColumn`` exceptions are propagated, like other errors
    raised by the transformer.

    .. note::

        The ``transform`` and ``fit_transform`` methods of ``transformer`` must
        return a column, a list of columns or a dataframe of the same module
        (polars or pandas) as the input, either by default or by supporting the
        scikit-learn ``set_output`` API.

    Parameters
    ----------
    transformer : scikit-learn Transformer
        The transformer to map to the selected columns. For each column in
        ``cols``, a clone of the transformer is created then ``fit_transform``
        is called on a single-column dataframe. If the transformer has a
        ``__single_column_transformer__`` attribute, ``fit_transform`` is
        passed directly the column (a pandas or polars Series) rather than a
        DataFrame. ``fit_transform`` must return either a DataFrame, a Series,
        or a list of Series. ``fit_transform`` can raise ``RejectColumn`` to
        indicate that this transformer does not apply to this column -- for
        example the ``ToDatetime`` transformer will raise ``RejectColumn`` for
        numerical columns. In this case, the column will appear unchanged in
        the output.

    cols : str, sequence of str, or skrub selector, optional
        The columns to attempt to transform. Columns outside of this selection
        will be passed through unchanged, without attempting to call
        ``fit_transform`` on them. The default is to attempt transforming all
        columns.

    allow_reject : bool, default=False
        Whether the transformer is allowed to reject a column by raising a
        ``RejectColumn`` exception. If ``True``, rejected columns will be
        passed through unchanged by ``OnEachColumn`` and will not appear in
        attributes such as ``transformers_``, ``used_inputs_``, etc. If
        ``False``, column rejections are considered as errors and
        ``RejectColumn`` exceptions are propagated.

    keep_original : bool, default=False
        If ``True``, the original columns are preserved in the output. If the
        transformer produces a column with the same name, the transformation
        result is renamed so that both columns can appear in the output. If
        ``False``, when the transformer accepts a column, only the
        transformer's output is included in the result, not the original
        column. In all cases rejected columns (or columns not selected by
        ``cols``) are passed through.

    rename_columns : str, default='{}'
        Format string applied to all transformation ouput column names. For
        example pass ``'transformed_{}'`` to prepend ``'transformed_'`` to all
        output column names. The default value does not modify the names.
        Renaming is not applied to columns not selected by ``cols``.

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

    output_to_input_ : dict
        Maps the name of each column in the transformed output to the name of
        the input column from which it was derived.

    transformers_ : dict
        Maps the name of each column that was transformed to the corresponding
        fitted transformer.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub._on_each_column import OnEachColumn
    >>> from sklearn.preprocessing import StandardScaler
    >>> df = pd.DataFrame(dict(A=[-10., 10.], B=[-10., 0.], C=[0., 10.]))
    >>> df
          A     B     C
    0 -10.0 -10.0   0.0
    1  10.0   0.0  10.0

    Fit a StandardScaler to each column in df:

    >>> scaler = OnEachColumn(StandardScaler())
    >>> scaler.fit_transform(df)
         A    B    C
    0 -1.0 -1.0 -1.0
    1  1.0  1.0  1.0
    >>> scaler.transformers_
    {'A': StandardScaler(), 'B': StandardScaler(), 'C': StandardScaler()}

    We can restrict the columns on which the transformation is applied:

    >>> scaler = OnEachColumn(StandardScaler(), cols=["A", "B"])
    >>> scaler.fit_transform(df)
         A    B     C
    0 -1.0 -1.0   0.0
    1  1.0  1.0  10.0

    We see that the scaling has not been applied to "C", which also does not
    appear in the transformers_:

    >>> scaler.transformers_
    {'A': StandardScaler(), 'B': StandardScaler()}
    >>> scaler.used_inputs_
    ['A', 'B']

    **Rejected columns**

    The transformer can raise ``RejectColumn`` to indicate it cannot handle a
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
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Could not find a datetime format for column 'city'.

    How these rejections are handled depends on the ``allow_reject`` parameter.
    By default, no special handling is performed and rejections are considered
    to be errors:

    >>> to_datetime = OnEachColumn(ToDatetime())
    >>> to_datetime.fit_transform(df)
    Traceback (most recent call last):
        ...
    ValueError: Transformer ToDatetime.fit_transform failed on column 'city'. See above for the full traceback.

    However, setting ``allow_reject=True`` gives the transformer itself some
    control over which columns it should be applied to. For example, whether a
    string column contains dates is only known once we try to parse them.
    Therefore it might be sensible to try to parse all string columns but allow
    the transformer to reject those that, upon inspection, do not contain dates.

    >>> to_datetime = OnEachColumn(ToDatetime(), allow_reject=True)
    >>> transformed = to_datetime.fit_transform(df)
    >>> transformed
        birthday    city
    0 2024-01-29  London

    Now the column 'city' was rejected but this was not treated as an error;
    'city' was passed through unchanged and only 'birthday' was converted to a
    datetime column.

    >>> transformed.dtypes
    birthday    datetime64[ns]
    city                object
    dtype: object
    >>> to_datetime.transformers_
    {'birthday': ToDatetime()}

    **Renaming outputs & keeping the original columns**

    The ``rename_columns`` parameter allows renaming output columns.

    >>> df = pd.DataFrame(dict(A=[-10., 10.], B=[0., 100.]))
    >>> scaler = OnEachColumn(StandardScaler(), rename_columns='{}_scaled')
    >>> scaler.fit_transform(df)
       A_scaled  B_scaled
    0      -1.0      -1.0
    1       1.0       1.0

    The renaming is only applied to columns selected by ``cols`` (and not
    rejected by the transformer when ``allow_reject`` is ``True``).

    >>> scaler = OnEachColumn(StandardScaler(), cols=['A'], rename_columns='{}_scaled')
    >>> scaler.fit_transform(df)
       A_scaled      B
    0      -1.0    0.0
    1       1.0  100.0

    ``rename_columns`` can be particularly useful when ``keep_original`` is
    ``True``. When a column is transformed, we can tell ``OnEachColumn`` to
    retain the original, untransformed column in the output. If the transformer
    produces a column with the same name, the transformation result is renamed
    to avoid a name clash.

    >>> scaler = OnEachColumn(StandardScaler(), keep_original=True)
    >>> scaler.fit_transform(df)                                    # doctest: +SKIP
          A  A__skrub_89725c56__      B  B__skrub_81cc7d00__
    0 -10.0                 -1.0    0.0                 -1.0
    1  10.0                  1.0  100.0                  1.0

    In this case we may want to set a more sensible name for the transformer's output:

    >>> scaler = OnEachColumn(
    ...     StandardScaler(), keep_original=True, rename_columns="{}_scaled"
    ... )
    >>> scaler.fit_transform(df)
          A  A_scaled      B  B_scaled
    0 -10.0      -1.0    0.0      -1.0
    1  10.0       1.0  100.0       1.0
    """  # noqa: E501

    def __init__(
        self,
        transformer,
        cols=_selectors.all(),
        allow_reject=False,
        keep_original=False,
        rename_columns="{}",
        n_jobs=None,
    ):
        self.transformer = transformer
        self.cols = cols
        self.allow_reject = allow_reject
        self.keep_original = keep_original
        self.rename_columns = rename_columns
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        self._columns = _selectors.make_selector(self.cols).expand(X)
        results = []
        all_columns = sbd.column_names(X)
        parallel = Parallel(n_jobs=self.n_jobs)
        func = delayed(_fit_transform_column)
        results = parallel(
            func(
                sbd.col(X, col_name),
                y,
                self._columns,
                self.transformer,
                self.allow_reject,
            )
            for col_name in all_columns
        )
        return self._process_fit_transform_results(results, X)

    def transform(self, X):
        check_is_fitted(self, "transformers_")
        parallel = Parallel(n_jobs=self.n_jobs)
        func = delayed(_transform_column)
        outputs = parallel(
            func(
                sbd.col(X, col_name),
                self.transformers_.get(col_name),
            )
            for col_name in sbd.column_names(X)
        )
        transformed_columns = []
        for col_name, col_outputs in zip(sbd.column_names(X), outputs):
            if self.transformers_.get(col_name) is not None and self.keep_original:
                col_outputs = [sbd.col(X, col_name)] + col_outputs
            transformed_columns.extend(col_outputs)
        transformed_columns = _rename_columns(transformed_columns, self.all_outputs_)
        result = sbd.make_dataframe_like(X, transformed_columns)
        result = sbd.copy_index(X, result)
        return result

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
                suggested_names = list(
                    map(_utils.renaming_func(self.rename_columns), suggested_names)
                )
                output_names = pick_column_names(
                    suggested_names,
                    forbidden_names - (set() if self.keep_original else {input_name}),
                )
                output_cols = _rename_columns(output_cols, output_names)
                forbidden_names.update(output_names)
                self.transformers_[input_name] = transformer
                self.input_to_outputs_[input_name] = output_names
                self.output_to_input_.update(**{o: input_name for o in output_names})
                if self.keep_original:
                    output_cols = [sbd.col(X, input_name)] + output_cols
            transformed_columns.extend(output_cols)

        self.all_outputs_ = _column_names(transformed_columns)
        self.used_inputs_ = list(self.transformers_.keys())
        self.created_outputs_ = list(itertools.chain(*self.input_to_outputs_.values()))
        # for sklearn
        self.feature_names_in_ = self.all_inputs_
        self.n_features_in_ = len(self.all_inputs_)

        result = sbd.make_dataframe_like(X, transformed_columns)
        result = sbd.copy_index(X, result)
        return result

    # set_output api compatibility

    def get_feature_names_out(self):
        return self.all_outputs_


def _prepare_transformer_input(transformer, column):
    if hasattr(transformer, "__single_column_transformer__"):
        return column
    return sbd.make_dataframe_like(column, [column])


def _fit_transform_column(column, y, columns_to_handle, transformer, allow_reject):
    col_name = sbd.name(column)
    if col_name not in columns_to_handle:
        return col_name, [column], None
    transformer = clone(transformer)
    _utils.set_output(transformer, column)
    transformer_input = _prepare_transformer_input(transformer, column)
    allowed = (RejectColumn,) if allow_reject else ()
    try:
        output = transformer.fit_transform(transformer_input, y=y)
    except allowed:
        return col_name, [column], None
    except Exception as e:
        raise ValueError(
            f"Transformer {transformer.__class__.__name__}.fit_transform "
            f"failed on column {col_name!r}. See above for the full traceback."
        ) from e
    output = _utils.check_output(transformer, transformer_input, output)
    output_cols = sbd.to_column_list(output)
    return col_name, output_cols, transformer


def _transform_column(column, transformer):
    if transformer is None:
        return [column]
    transformer_input = _prepare_transformer_input(transformer, column)
    try:
        output = transformer.transform(transformer_input)
    except Exception as e:
        raise ValueError(
            f"Transformer {transformer.__class__.__name__}.transform "
            f"failed on column {sbd.name(column)!r}. See above for the full traceback."
        ) from e
    # we do not call `_utils.check_output` here, assuming that if the output
    # had a correct type (e.g. polars dataframe) in `fit_transform` it will
    # have the same (correct) type in `transform`.
    output_cols = sbd.to_column_list(output)
    return output_cols


def _column_names(column_list):
    return [sbd.name(column) for column in column_list]


def _rename_columns(columns_list, new_names):
    return [sbd.rename(column, name) for (column, name) in zip(columns_list, new_names)]
