import reprlib
from collections import UserDict

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
from ._on_each_column import SingleColumnTransformer
from ._select_cols import Drop
from ._to_datetime import ToDatetime
from ._to_float32 import ToFloat32
from ._to_str import ToStr
from ._wrap_transformer import wrap_transformer


class PassThrough(SingleColumnTransformer):
    def fit_transform(self, column):
        return column

    def transform(self, column):
        return column


HIGH_CARDINALITY_TRANSFORMER = GapEncoder(n_components=30)
LOW_CARDINALITY_TRANSFORMER = OneHotEncoder(
    sparse_output=False,
    dtype="float32",
    handle_unknown="ignore",
    drop="if_binary",
)
DATETIME_TRANSFORMER = EncodeDatetime()
NUMERIC_TRANSFORMER = PassThrough()


class ShortReprDict(UserDict):
    """A dict with a shorter repr.

    Examples
    --------
    >>> d = {'one': 1, 'two': 2, 'three': 3, 'four': 4}
    >>> d
    {'one': 1, 'two': 2, 'three': 3, 'four': 4}
    >>> from skrub._table_vectorizer import ShortReprDict
    >>> ShortReprDict(d)
    {'four': 4, 'one': 1, ...}
    >>> _['two']
    2
    """

    def __repr__(self):
        r = reprlib.Repr()
        r.maxdict = 2
        return r.repr(dict(self))


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


class TableVectorizer(TransformerMixin, BaseEstimator):
    """Transform a dataframe to a numerical (vectorized) representation.

    Applies a different transformation to each of several kinds of columns:

    - numeric:
        Floats, ints, and Booleans.
    - datetime:
        Datetimes and dates.
    - low_cardinality:
        String and categorical columns with a count of unique values smaller
        than a given threshold (40 by default). Category encoding schemes such
        as one-hot encoding, ordinal encoding etc. are typically appropriate
        for low-cardinality columns.
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
    overriding the categorization described above. This is done by providing a
    list of pairs ``(transformer, list_of_columns)`` as the
    ``specific_transformers`` parameter.

    .. note::

        The ``specific_transformers`` parameter is likely to be removed in a
        future version of ``skrub``, when better utilities for building complex
        pipelines are introduced.

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
        The transformer for ``numeric`` columns. The default is passthrough.

    datetime_transformer : transformer, "passthrough" or "drop", optional
        The transformer for ``datetime`` columns. The default is
        ``EncodeDatetime``, which extracts features such as year, month, etc.

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
    transformers_ : dict
        Maps the name of each column to the main transformer (such as the
        ``high_cardinality_transformer``, as opposed to the preprocessing
        steps) that was fitted on it.
    all_processing_steps_ : dict
        Maps the name of each column to a list of all the processing steps that
        were applied to it. Those steps may include some pre-processing
        transformations such as converting strings to datetimes or numbers, the
        main transformer, and casting the main transformer's output to float32.
    input_to_outputs_ : dict
        Maps the name of each column that was transformed by the
        TableVectorizer to the names of the corresponding output columns.
    output_to_input_ : dict
        Maps the name of each output column to the name of the column in the
        input dataframe from which it was derived.
    feature_names_in_ : list of strings
        The names of the input columns, after applying some cleaning (casting
        all column names to strings and deduplication).
    all_outputs_ : list of strings
        The names of the output columns.

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
    >>> df.dtypes
    A    object
    B    object
    C    object
    dtype: object

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

    We can also see the encoder, i.e. the main transformer that was applied to
    a given column:

    >>> vectorizer.transformers_['B']
    EncodeDatetime()
    >>> vectorizer.transformers_['A']
    OneHotEncoder(drop='if_binary', dtype='float32', handle_unknown='ignore',
                  sparse_output=False)
    >>> vectorizer.transformers_['A'].categories_
    [array(['one', 'three', 'two'], dtype=object)]

    We can see the columns grouped by the kind of encoder that was applied
    to them.

    >>> vectorizer.column_kinds_
    {'numeric': ['C'], 'datetime': ['B'], 'low_cardinality': ['A'], 'high_cardinality': [], 'specific': []}

    Before applying the main transformer, the ``TableVectorizer`` applies
    several preprocessing steps, for example to detect numbers or dates that are
    represented as strings. Moreover, a final post-processing step is applied to
    all non-categorical columns in the encoder's output to cast them to float32.
    We can inspect all the processing steps that were applied to a given column:

    >>> vectorizer.all_processing_steps_['B']
    [CleanNullStrings(), ToDatetime(), EncodeDatetime(), {'B_day': ToFloat32(), 'B_month': ToFloat32(), ...}]

    Note that as the encoder (``EncodeDatetime()`` above) produces multiple
    columns, the last processing step is not described by a single transformer
    like the previous ones but by a mapping from column name to transformer.

    ``all_processing_steps_`` is useful to inspect the details of the
    choices made by the ``TableVectorizer`` during preprocessing, for example:

    >>> vectorizer.all_processing_steps_['B'][1]
    ToDatetime()
    >>> _.datetime_format_
    '%d/%m/%Y'

    **Transformers are applied separately to each column**

    The ``TableVectorizer`` vectorizes each column separately -- a different
    transformer is applied to each column; multivariate transformers are not
    allowed.

    >>> df_1 = pd.DataFrame(dict(A=['one', 'two'], B=['three', 'four']))
    >>> vectorizer = TableVectorizer().fit(df_1)
    >>> vectorizer.transformers_['A'] is not vectorizer.transformers_['B']
    True
    >>> vectorizer.transformers_['A'].categories_
    [array(['one', 'two'], dtype=object)]
    >>> vectorizer.transformers_['B'].categories_
    [array(['four', 'three'], dtype=object)]

    **Overriding the transformer for specific columns**

    We can also provide transformers for specific columns. In that case the
    provided transformer has full control over the associated columns; no other
    processing is applied to those columns. A column cannot appear twice in the
    ``specific_transformers``.

    .. note::

        This functionality is likely to be removed in a future version of the
        ``TableVectorizer``.

    The overrides are provided as a list of pairs:
    ``(transformer, list_of_column_names)``.

    >>> from sklearn.preprocessing import OrdinalEncoder
    >>> vectorizer = TableVectorizer(
    ...     specific_transformers=[('drop', ['A']), (OrdinalEncoder(), ['B'])]
    ... )
    >>> df
           A           B     C
    0    one  02/02/2024   1.5
    1    two  23/02/2024   N/A
    2    two  12/03/2024  12.2
    3  three  13/03/2024   N/A
    >>> vectorizer.fit_transform(df)
         B     C
    0  0.0   1.5
    1  3.0   NaN
    2  1.0  12.2
    3  2.0   NaN

    Here the column 'A' has been dropped and the column 'B' has been passed to
    the ``OrdinalEncoder`` (instead of the default choice which would have been
    ``DatetimeEncoder``).

    We can see that 'A' and 'B' are now treated as 'specific' columns:

    >>> vectorizer.column_kinds_
    {'numeric': ['C'], 'datetime': [], 'low_cardinality': [], 'high_cardinality': [], 'specific': ['A', 'B']}

    Preprocessing and postprocessing steps are not applied to columns appearing
    in ``specific_columns``. For example 'B' has not gone through
    ``ToDatetime()``:

    >>> vectorizer.all_processing_steps_
    {'A': [Drop()], 'B': [OrdinalEncoder()], 'C': [CleanNullStrings(), ToFloat32(), PassThrough(), {'C': ToFloat32()}]}

    Specifying several ``specific_transformers`` for the same column is not allowed.

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
        self.fit_transform(X, y=y)
        return self

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
        self._check_specific_columns()
        self._make_pipeline()
        output = self._pipeline.fit_transform(X, y=y)
        self.feature_names_in_ = self._preprocessors[0].feature_names_out_
        self.all_outputs_ = sbd.column_names(output)
        self._store_processing_steps()
        self._store_column_kinds()
        self._store_output_to_input()
        return output

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

    def _check_specific_columns(self):
        specific_columns = {}
        for i, config in enumerate(self.specific_transformers):
            try:
                _, cols = config
            except (ValueError, TypeError):
                raise ValueError(
                    "Expected a list of (transformer, columns) pairs. "
                    f"Got {config!r} at index {i}."
                )
            for c in cols:
                if not isinstance(c, str):
                    raise ValueError(f"cols must be string names, got {c}")
                if c in specific_columns:
                    raise ValueError(
                        f"Column {c!r} used twice in in specific_transformers, "
                        f"at indices {specific_columns[c]} and {i}."
                    )
            specific_columns |= {c: i for c in cols}
        self._specific_columns = list(specific_columns.keys())

    def _make_pipeline(self):
        def add_step(steps, transformer, cols, allow_reject=False):
            steps.append(
                wrap_transformer(
                    _check_transformer(transformer),
                    cols,
                    allow_reject=allow_reject,
                    n_jobs=self.n_jobs,
                    columnwise=True,
                )
            )
            return steps[-1]

        cols = s.all() - self._specific_columns

        self._preprocessors = [CheckInputDataFrame()]
        for transformer in [
            CleanNullStrings(),
            ToDatetime(),
            ToFloat32(),
            CleanCategories(),
            ToStr(),
        ]:
            add_step(self._preprocessors, transformer, cols, allow_reject=True)

        self._encoders = []
        self._named_encoders = {}
        for name, selector in [
            ("numeric", s.numeric()),
            ("datetime", s.any_date()),
            (
                "low_cardinality",
                s.cardinality_below(self.cardinality_threshold),
            ),
            ("high_cardinality", s.all()),
        ]:
            self._named_encoders[name] = add_step(
                self._encoders,
                getattr(self, f"{name}_transformer"),
                cols & selector - created_by(*self._encoders),
            )

        self._specific_transformers = []
        for specific_transformer, specific_cols in self.specific_transformers:
            add_step(self._specific_transformers, specific_transformer, specific_cols)

        self._postprocessors = []
        add_step(
            self._postprocessors,
            ToFloat32(),
            s.all() - created_by(*self._specific_transformers) - s.categorical(),
            allow_reject=True,
        )
        self._pipeline = make_pipeline(
            *self._preprocessors,
            *self._encoders,
            *self._specific_transformers,
            *self._postprocessors,
        )

    def _store_processing_steps(self):
        input_names = self._preprocessors[0].feature_names_out_
        to_outputs = {col: [col] for col in input_names}
        to_steps = {col: [] for col in input_names}
        self.transformers_ = {}
        # [1:] because CheckInputDataFrame not included in all_processing_steps_
        for step in self._preprocessors[1:]:
            for col, transformer in step.transformers_.items():
                to_steps[col].append(transformer)
        for step in self._encoders + self._specific_transformers:
            for col, transformer in step.transformers_.items():
                to_steps[col].append(transformer)
                to_outputs[col] = step.input_to_outputs_[col]
                self.transformers_[col] = transformer
        for col, outputs in to_outputs.items():
            post_proc = {
                c: t
                for c in outputs
                if (t := self._postprocessors[0].transformers_.get(c)) is not None
            }
            if post_proc:
                to_steps[col].append(ShortReprDict(post_proc))
        self.input_to_outputs_ = to_outputs
        self.all_processing_steps_ = to_steps

    def _store_column_kinds(self):
        self.column_kinds_ = {
            k: v.used_inputs_ for k, v in self._named_encoders.items()
        }
        self.column_kinds_["specific"] = self._specific_columns

    def _store_output_to_input(self):
        self.output_to_input_ = {
            out: input_
            for (input_, outputs) in self.input_to_outputs_.items()
            for out in outputs
        }

    # scikt-learn compatibility

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
        check_is_fitted(self, "all_outputs_")
        return np.asarray(self.all_outputs_)
