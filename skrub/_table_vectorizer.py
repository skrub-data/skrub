import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from . import _selectors as s
from . import _utils
from ._check_input import CheckInputDataFrame
from ._clean_categories import CleanCategories
from ._clean_null_strings import CleanNullStrings
from ._datetime_encoder import EncodeDatetime
from ._gap_encoder import GapEncoder
from ._select_cols import Drop
from ._to_datetime import ToDatetime
from ._to_float import ToFloat
from ._to_str import ToStr
from ._wrap_transformer import wrap_transformer


class PassThrough(BaseEstimator):
    __single_column_transformer__ = True

    def fit_transform(self, column):
        return column

    def transform(self, column):
        return column

    def fit(self, column):
        self.fit_transform(column)
        return self


HIGH_CARDINALITY_TRANSFORMER = GapEncoder(n_components=30)
LOW_CARDINALITY_TRANSFORMER = OneHotEncoder(
    sparse_output=False,
    dtype="float32",
    handle_unknown="ignore",
    drop="if_binary",
)
DATETIME_TRANSFORMER = EncodeDatetime()
NUMERIC_TRANSFORMER = PassThrough()


def _created_by(col, transformers):
    return any(sbd.name(col) in t.created_outputs_ for t in transformers)


def created_by(*transformers):
    return s.Filter(
        _created_by,
        args=(transformers,),
        selector_repr=f"created_by(<any of {len(transformers)} transformers>)",
    )


def _check_transformer(transformer):
    if isinstance(transformer, str):
        if transformer == "passthrough":
            return PassThrough()
        if transformer == "drop":
            return Drop()
        raise ValueError(
            f"Value not understood: {transformer!r}. Please provide either"
            " 'passthrough', 'drop', or a scikit-learn transformer."
        )
    return clone(transformer)


class TableVectorizer(TransformerMixin, BaseEstimator, auto_wrap_output_keys=()):
    """Transform a dataframe to a numerical array.

    Applies a different transformation to each of several kinds of columns:

    - numeric:
        Floats ints, and Booleans.
    - datetime:
        Datetimes and dates.
    - low_cardinality:
        String and categorical columns with a count of unique values smaller
        than a given threshold (40 by default). Category encoding schemes such
        as one-hot encoding, ordinal encoding etc. are typically appropriate
        for low_cardinality columns.
    - high_cardinality:
        String and categorical columns with many unique values, such as
        free-form text. Such columns have so many distinct values that it is
        not possible to assign a distinct representation to each: the dimension
        would be too large and there would be too few examples of each
        category. Representations designed for text, such as topic modelling
        (GapEncoder) or locality-sensitive hashing (MinHash) are more
        appropriate.

    .. note::

        Transformations are applied **independently on each column**. A
        different transformer instance is used for each column separately;
        multivariate transformations are therefore not supported.

    The transformer for each kind of column can be configured with the
    corresponding ``*_transformer`` parameter: ``numeric_transformer``,
    ``datetime_transformer``, ...

    A transformer can be a scikit-learn Transformer (an object providing the
    ``fit``, ``fit_transform`` and ``transform`` methods), a clone of which
    will be applied to each column separately. A transformer can also be the
    literal string ``"drop"`` to drop the corresponding columns (they will not
    appear in the output), or ``"passthrough"`` to leave them unchanged.

    Additionally, it is possible to specify transformers for specific columns,
    overriding the categories described above. This is done by providing pairs
    of (transformer, list_of_columns) as the ``specific_transformers``
    parameter.

    Parameters
    ----------
    cardinality_threshold : int, default=40
        String and categorical features with a number of unique values strictly
        smaller than this threshold are considered ``low_cardinality``, the
        rest are considered ``high_cardinality``.

    low_cardinality_transformer : transformer, "passthrough" or "drop", optional
        The transformer for ``low_cardinality`` columns. The default is a
        ``OneHotEncoder``.

    high_cardinality_transformer : transformer, "passthrough" or "drop", optional
        The transformer for ``high_cardinality`` columns. The default is a
        ``GapEncoder`` with 30 components (30 output columns for each input).

    numeric_transformer : transformer, "passthrough" or "drop", optional
        The transformer for ``numeric`` columns. The default simply casts
        numerical columns to ``Float32``.

    datetime_transformer : transformer, "passthrough" or "drop", optional
        The transformer for ``datetime`` columns. The default is a
        ``DatetimeEncoder``.

    specific_transformers : list of (transformer, list of column names) pairs, optional
        Override the categories above for the given columns and force using the
        specified transformer. This disables any preprocessing usually done by
        the TableVectorizer; the columns are passed to the transformer without
        any modification. Using ``specific_transformers`` provides similar
        functionality to what is offered by scikit-learn's
        ``ColumnTransformer``.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a joblib ``parallel_backend`` context.
        ``-1`` means using all processors.

    Attributes
    ----------
    pipeline_ : scikit-learn Pipeline
        The TableVectorizer, under the hood, is just a scikit-learn pipeline.
        This attribute exposes the fitted pipeline. It is also possible to
        obtain an equivalent pipeline without fitting the TableVectorizer by
        calling ``make_pipeline``, in order to chain the steps it implements
        with the rest of a bigger pipeline rather than nesting them inside a
        TableVectorizer step.
    input_to_outputs_ : dict
        Maps the name of each column that was transformed by the
        TableVectorizer to the names of the corresponding output columns. The
        columns specified in the ``passthrough`` parameter are not included.
    output_to_input_ : dict
        Maps the name of each output column to the name of the column in the
        input dataframe from which it was derived.
    all_outputs_ :
        The names of all output columns, including those that are passed
        through unchanged.

    Examples
    --------
    >>> from skrub import TableVectorizer
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': ['one', 'two', 'two', 'three'],
    ...     'B': ['02/02/2024', '23/02/2024', '12/03/2024', '13/03/2024'],
    ...     'C': ['1.5', 'N/A', '12.2', 'N/A'],
    ... })
    >>> df
           A           B     C
    0    one  02/02/2024   1.5
    1    two  23/02/2024   N/A
    2    two  12/03/2024  12.2
    3  three  13/03/2024   N/A

    >>> vectorizer = TableVectorizer()
    >>> vectorizer.fit_transform(df)
       A_one  A_three  A_two  B_year  B_month  B_day  B_total_seconds     C
    0    1.0      0.0    0.0  2024.0      2.0    2.0     1.706832e+09   1.5
    1    0.0      0.0    1.0  2024.0      2.0   23.0     1.708646e+09   NaN
    2    0.0      0.0    1.0  2024.0      3.0   12.0     1.710202e+09  12.2
    3    0.0      1.0    0.0  2024.0      3.0   13.0     1.710288e+09   NaN

    We can inspect which outputs were created from a given column in the input
    dataframe:

    >>> vectorizer.input_to_outputs_['B']
    ['B_year', 'B_month', 'B_day', 'B_total_seconds']

    and the reverse mapping:

    >>> vectorizer.output_to_input_['B_total_seconds']
    'B'

    We can also see the final transformer that was applied to a given column:

    >>> vectorizer.transformers_['A']
    OneHotEncoder(drop='if_binary', dtype='float32', handle_unknown='ignore',
                  sparse_output=False)
    >>> vectorizer.transformers_['A'].categories_
    [array(['one', 'three', 'two'], dtype=object)]

    Before applying that final transformer, the ``TableVectorizer`` applies
    several preprocessing steps to all columns, for example to detect numbers
    or dates that are represented as strings. We can inspect all the processing
    steps that were applied to a given column:

    >>> vectorizer.input_to_processing_steps_['B']
    [CleanNullStrings(), ToDatetime(), EncodeDatetime()]

    >>> vectorizer.transformers_['B']
    EncodeDatetime()

    **Transformers are applied separately to each column**

    The ``TableVectorizer`` vectorizes each column separately -- a different
    transformer is applied to each column; multivariate transformers are not
    allowed.

    >>> df = pd.DataFrame(dict(A=['one', 'two'], B=['three', 'four']))
    >>> vectorizer = TableVectorizer().fit(df)
    >>> vectorizer.transformers_['A'] is not vectorizer.transformers_['B']
    True
    >>> vectorizer.transformers_['A'].categories_
    [array(['one', 'two'], dtype=object)]
    >>> vectorizer.transformers_['B'].categories_
    [array(['four', 'three'], dtype=object)]

    **Overriding the transformer for specific columns**

    We can also provide transformers for specific columns

    >>> from sklearn.preprocessing import OneHotEncoder
    >>> ohe = OneHotEncoder(sparse_output=False)
    >>> vectorizer = TableVectorizer(
    ...     specific_transformers=[('drop', ['A']), (ohe, ['B'])]
    ... )
    >>> vectorizer.fit_transform(df)
       B_four  B_three
    0     0.0      1.0
    1     1.0      0.0

    Here the column 'A' has been dropped and the column 'B' has been passed to
    the ``OneHotEncoder`` (without any preprocessing such as converting it to
    numbers as was done in the first example).

    When a column is used in one of the ``specific_transformers``, none of the
    preprocessing steps are applied and the specific transformer controls the
    full transformation for that column.

    >>> df = pd.DataFrame(dict(A=['1.1', '2.2', 'N/A'], B=['1.1', '2.2', 'N/A']))
    >>> df
         A    B
    0  1.1  1.1
    1  2.2  2.2
    2  N/A  N/A
    >>> vectorizer = TableVectorizer(
    ...     numeric_transformer='passthrough',
    ...     specific_transformers=[('passthrough', ['B'])],
    ... )
    >>> vectorizer.fit_transform(df)
         A    B
    0  1.1  1.1
    1  2.2  2.2
    2  NaN  N/A

    >>> vectorizer.input_to_processing_steps_
    {'A': [CleanNullStrings(), ToFloat(), PassThrough()], 'B': [PassThrough()]}

    Here we can see that the final estimator for both columns is passthrough,
    but unlike ``'B'``, ``'A'`` went through the default preprocessing steps
    and was converted to a numeric column.

    Specifying several specific transformers for the same column is not allowed.

    >>> vectorizer = TableVectorizer(
    ...     specific_transformers=[('passthrough', ['A', 'B']), ('drop', ['A'])]
    ... )

    >>> vectorizer.fit_transform(df)
    Traceback (most recent call last):
        ...
    ValueError: Column 'A' used twice in in specific_transformers, at indices 0 and 1.
    """

    def __init__(
        self,
        *,
        cardinality_threshold=40,
        low_cardinality_transformer=LOW_CARDINALITY_TRANSFORMER,
        high_cardinality_transformer=HIGH_CARDINALITY_TRANSFORMER,
        numeric_transformer=NUMERIC_TRANSFORMER,
        datetime_transformer=DATETIME_TRANSFORMER,
        specific_transformers=(),
        n_jobs=None,
    ):
        self.cardinality_threshold = cardinality_threshold
        self.low_cardinality_transformer = _utils.clone_if_default(
            low_cardinality_transformer, LOW_CARDINALITY_TRANSFORMER
        )
        self.high_cardinality_transformer = _utils.clone_if_default(
            high_cardinality_transformer, HIGH_CARDINALITY_TRANSFORMER
        )
        self.numeric_transformer = _utils.clone_if_default(
            numeric_transformer, NUMERIC_TRANSFORMER
        )
        self.datetime_transformer = _utils.clone_if_default(
            datetime_transformer, DATETIME_TRANSFORMER
        )
        self.specific_transformers = specific_transformers
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """Fit transformer.

        Parameters
        ----------
        X : dataframe
            Input data to transform.

        y : any type, default=None
            This parameter exists for compatibility with the scikit-learn API
            and is ignored.

        Returns
        -------
        self : TableVectorizer
            This estimator.
        """
        self.fit_transform(X)
        return self

    def _make_pipeline(self):
        def add_step(steps, transformer, cols):
            steps.append(
                wrap_transformer(
                    _check_transformer(transformer),
                    cols,
                    n_jobs=self.n_jobs,
                    columnwise=True,
                )
            )

        cols = s.all() - self._user_managed_columns

        cleaning_steps = [CheckInputDataFrame()]
        for transformer in [
            CleanNullStrings(),
            ToDatetime(),
            ToFloat(),
            CleanCategories(),
            ToStr(),
        ]:
            add_step(cleaning_steps, transformer, cols)

        encoding_steps = []
        for transformer, selector in [
            (self.numeric_transformer, s.numeric()),
            (self.datetime_transformer, s.any_date()),
            (
                self.low_cardinality_transformer,
                s.cardinality_below(self.cardinality_threshold),
            ),
            (self.high_cardinality_transformer, s.all()),
        ]:
            add_step(
                encoding_steps,
                transformer,
                cols & selector - created_by(*encoding_steps),
            )

        user_steps = []
        for user_transformer, user_cols in self.specific_transformers_:
            add_step(user_steps, user_transformer, user_cols)

        self._pipeline = make_pipeline(*cleaning_steps, *encoding_steps, *user_steps)

    def fit_transform(self, X, y=None):
        """Fit transformer and transform dataframe.

        Parameters
        ----------
        X : dataframe
            Input data to transform.

        y : any type, default=None
            This parameter exists for compatibility with the scikit-learn API
            and is ignored.

        Returns
        -------
        dataframe
            The transformed input.
        """
        self._check_specific_transformers()
        self._make_pipeline()
        output = self._pipeline.fit_transform(X)
        self.feature_names_in_ = self._pipeline.steps[0][1].feature_names_out_
        self.all_outputs_ = sbd.column_names(output)
        self._store_input_transformations()
        self._store_processed_cols()
        self._store_transformers()
        self._store_output_to_input()
        return output

    def _check_specific_transformers(self):
        user_managed_columns = {}
        self.specific_transformers_ = []
        for i, config in enumerate(self.specific_transformers):
            try:
                transformer, cols = config
            except (ValueError, TypeError):
                raise ValueError(
                    "Expected a list of (transformer, columns) pairs. "
                    f"Got {config!r} at index {i}."
                )
            for c in cols:
                if not isinstance(c, str):
                    raise ValueError(f"cols must be string names, got {c}")
                if c in user_managed_columns:
                    raise ValueError(
                        f"Column {c!r} used twice in in specific_transformers, "
                        f"at indices {user_managed_columns[c]} and {i}."
                    )
            user_managed_columns |= {c: i for c in cols}
            self.specific_transformers_.append((_check_transformer(transformer), cols))
        self._user_managed_columns = list(user_managed_columns.keys())

    def transform(self, X):
        """Transform dataframe.

        Parameters
        ----------
        X : dataframe
            Input data to transform.

        Returns
        -------
        dataframe
            The transformed input.
        """
        return self._pipeline.transform(X)

    def _more_tags(self) -> dict:
        """
        Used internally by sklearn to ease the estimator checks.
        """
        return {
            "X_types": ["2darray", "string"],
            "allow_nan": [True],
            "_xfail_checks": {
                "check_complex_data": "Passthrough complex columns as-is.",
            },
        }

    def get_feature_names_out(self):
        """Return the column names of the output of ``transform`` as a list of strings.

        Returns
        -------
        list of strings
            The column names.
        """
        """"""
        check_is_fitted(self, "all_outputs_")
        return np.asarray(self.all_outputs_)

    def _store_input_transformations(self):
        pipeline_steps = list(self._pipeline.named_steps.values())
        to_outputs = {col: [col] for col in pipeline_steps[0].feature_names_out_}
        to_steps = {col: [] for col in pipeline_steps[0].feature_names_out_}
        for step in pipeline_steps[1:]:
            for col, outputs_at_previous_step in to_outputs.items():
                new_outputs = []
                for output in outputs_at_previous_step:
                    new_outputs.extend(step.input_to_outputs_.get(output, [output]))
                    if hasattr(step, "transformers_") and output in step.transformers_:
                        to_steps[col].append(step.transformers_[output])
                    elif hasattr(step, "transformer_") and output in step.used_inputs_:
                        to_steps[col].append(step.transformer_)
                to_outputs[col] = new_outputs
        self.input_to_outputs_ = to_outputs
        self.input_to_processing_steps_ = to_steps

    def _store_processed_cols(self):
        self.used_inputs_, self.created_outputs_ = [], []
        for col in self.input_to_processing_steps_:
            if self.input_to_processing_steps_[col]:
                self.used_inputs_.append(col)
                self.created_outputs_.extend(self.input_to_outputs_[col])

    def _store_transformers(self):
        self.transformers_ = {
            c: ([None] + steps)[-1]
            for c, steps in self.input_to_processing_steps_.items()
        }

    def _store_output_to_input(self):
        self.output_to_input_ = {
            out: input_
            for (input_, outputs) in self.input_to_outputs_.items()
            for out in outputs
        }
