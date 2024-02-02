import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from . import _selectors as sbs
from . import _utils
from ._check_input import CheckInputDataFrame
from ._clean_null_strings import CleanNullStrings
from ._datetime_encoder import DatetimeColumnEncoder
from ._gap_encoder import GapEncoder
from ._map import Map
from ._pandas_convert_dtypes import PandasConvertDTypes
from ._to_categorical import ToCategorical
from ._to_datetime import ToDatetime
from ._to_float import ToFloat32
from ._to_numeric import ToNumeric

HIGH_CARDINALITY_TRANSFORMER = GapEncoder(n_components=30)
LOW_CARDINALITY_TRANSFORMER = OneHotEncoder(
    sparse_output=False,
    dtype="float32",
    handle_unknown="ignore",
    drop="if_binary",
)
DATETIME_TRANSFORMER = DatetimeColumnEncoder()
NUMERIC_TRANSFORMER = ToFloat32()


def _make_table_vectorizer_pipeline(
    low_cardinality_transformer,
    high_cardinality_transformer,
    numeric_transformer,
    datetime_transformer,
    remainder_transformer,
    cardinality_threshold,
    passthrough,
    n_jobs,
):
    cols = sbs.inv(passthrough)

    cleaning_steps = [
        ("check_input", CheckInputDataFrame()),
        ("convert_dtypes", Map(PandasConvertDTypes(), cols, n_jobs=n_jobs)),
        ("clean_null_strings", Map(CleanNullStrings(), cols, n_jobs=n_jobs)),
        ("to_datetime", Map(ToDatetime(), cols, n_jobs=n_jobs)),
        ("to_numeric", Map(ToNumeric(), cols, n_jobs=n_jobs)),
        (
            "to_categorical",
            Map(ToCategorical(cardinality_threshold - 1), cols, n_jobs=n_jobs),
        ),
    ]

    low_card_cat = sbs.categorical() & sbs.cardinality_below(cardinality_threshold)
    feature_extractors = [
        ("low_cardinality_transformer", low_cardinality_transformer, low_card_cat),
        ("high_cardinality_transformer", high_cardinality_transformer, sbs.string()),
        ("numeric_transformer", numeric_transformer, sbs.numeric() | sbs.boolean()),
        ("datetime_transformer", datetime_transformer, sbs.anydate()),
        ("remainder_transformer", remainder_transformer, sbs.all()),
    ]
    feature_extraction_steps = []
    for _, transformer, selector in feature_extractors:
        selector = (cols - sbs.produced_by(*feature_extraction_steps)) & selector
        feature_extraction_steps.append(Map(transformer, selector, n_jobs=n_jobs))
    feature_extraction_steps = [
        (name, step)
        for ((name, *_), step) in zip(feature_extractors, feature_extraction_steps)
    ]
    all_steps = cleaning_steps + feature_extraction_steps

    return Pipeline(all_steps)


class PassThrough(BaseEstimator):
    __single_column_transformer__ = True

    def fit_transform(self, column):
        return column

    def transform(self, column):
        return column

    def fit(self, column):
        self.fit_transform(column)
        return self


class Drop(BaseEstimator):
    __single_column_transformer__ = True

    def fit_transform(self, column):
        return []

    def transform(self, column):
        return []

    def fit(self, column):
        self.fit_transform(column)
        return self


def _clone_or_create_transformer(transformer):
    if isinstance(transformer, str):
        if transformer == "passthrough":
            return PassThrough()
        if transformer == "drop":
            return Drop()
        raise ValueError(
            f"Value not understood: {transformer!r}. Please provide either"
            " 'passthrough' or a scikit-learn transformer."
        )
    return clone(transformer)


class TableVectorizer(TransformerMixin, BaseEstimator, auto_wrap_output_keys=()):
    """Transform a dataframe to a numerical array.

    Applies a different transformation to each of several kinds of columns:

    - numeric:
        Floats and ints.
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
    - remainder:
        Columns that do not fall into any of the above categories (most likely
        with ``object`` dtype) are called the "remainder" columns and a
        different transformer can be specified for those.

    The transformer for each kind of column can be configured with the
    corresponding ``*_transformer`` parameter: ``numeric_transformer``,
    ``datetime_transformer``, ..., ``remainder_transformer``.

    A transformer can be a scikit-learn Transformer (an object providing the
    ``fit``, ``fit_transform`` and ``transform`` methods), a clone of which
    will be applied to each column separately. A transformer can also be the
    literal string ``"drop"`` to drop the corresponding columns (they will not
    appear in the output), or ``"passthrough"`` to leave them unchanged.

    Finally, it is possible to indicate that some columns should not be
    modified by the ``TableVectorizer``, and should be "passed through"
    unchanged (for example if we know they are already correctly encoded or we
    want to deal with them further down the data processing pipeline). The
    names of these columns must be given in the ``passthrough`` argument.

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

    remainder_transformer : transformer, "passthrough" or "drop", optional
        The transformer for ``remainder`` columns. To ensure that by default
        the output of the TableVectorizer is numerical that is valid input for
        scikit-learn estimators, the default for ``remainder_transformer`` is
        ``"drop"``.

    passthrough : str, sequence of str, or skrub selector, optional
        Columns to pass through without modifying them. Default is ``()``: all
        columns may be transformed.

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
    all_outputs_ :
        The names of all output columns, including those that are passed
        through unchanged.

    Examples
    --------
    >>> from skrub import TableVectorizer
    ... import pandas as pd
    ...
    ... df = pd.DataFrame(
    ...     {
    ...         "A": ["one", "two", "two", "three"],
    ...         "B": ["02/02/2024", "23/02/2024", "12/03/2024", "13/03/2024"],
    ...         "C": ["1.5", "N/A", "12.2", "N/A"],
    ...     }
    ... )
    ... TableVectorizer().fit_transform(df)
       A_one  A_three  A_two  B_year  B_month  B_day  B_total_seconds     C
    0    1.0      0.0    0.0  2024.0      2.0    2.0     1.706832e+09   1.5
    1    0.0      0.0    1.0  2024.0      2.0   23.0     1.708646e+09   NaN
    2    0.0      0.0    1.0  2024.0      3.0   12.0     1.710202e+09  12.2
    3    0.0      1.0    0.0  2024.0      3.0   13.0     1.710288e+09   NaN
    """

    def __init__(
        self,
        *,
        cardinality_threshold=40,
        low_cardinality_transformer=LOW_CARDINALITY_TRANSFORMER,
        high_cardinality_transformer=HIGH_CARDINALITY_TRANSFORMER,
        numeric_transformer=NUMERIC_TRANSFORMER,
        datetime_transformer=DATETIME_TRANSFORMER,
        remainder_transformer="drop",
        passthrough=(),
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
        self.remainder_transformer = remainder_transformer
        self.passthrough = passthrough
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

    def make_pipeline(self):
        """Make a scikit-learn pipeline that is equivalent to this transformer.

        Returns
        -------
        Pipeline
           An unfitted scikit-learn pipeline that is equivalent to this transformer.
        """
        return _make_table_vectorizer_pipeline(
            _clone_or_create_transformer(self.low_cardinality_transformer),
            _clone_or_create_transformer(self.high_cardinality_transformer),
            _clone_or_create_transformer(self.numeric_transformer),
            _clone_or_create_transformer(self.datetime_transformer),
            _clone_or_create_transformer(self.remainder_transformer),
            self.cardinality_threshold,
            self.passthrough,
            self.n_jobs,
        )

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
        self.pipeline_ = self.make_pipeline()
        output = self.pipeline_.fit_transform(X)
        self.feature_names_in_ = self.pipeline_.named_steps[
            "check_input"
        ].feature_names_in_
        self.all_outputs_ = sbd.column_names(output)
        self.input_to_outputs_ = _get_input_to_outputs_mapping(
            list(self.pipeline_.named_steps.values())[1:]
        )
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
        return self.pipeline_.transform(X)

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

    def get_processing_steps(self, kind=None):
        """Get all the processing steps applied to different columns.

        Parameters
        ----------
        kind : "numeric", "datetime", "high_cardinality", \
"low_cardinality", "remainder", or None, optional
            Filter results to return only those corresponding to the provided kind.
            None (the default) means return all results.

        Returns
        -------
        dict
            Mapping each column name to a list of (step_name, Transformer) pairs.
        """
        allowed_kinds = [
            "datetime",
            "numeric",
            "high_cardinality",
            "low_cardinality",
            "remainder",
            None,
        ]
        if kind not in allowed_kinds:
            raise ValueError(f"'kind' must be one of {allowed_kinds}. Got {kind!r}")
        col_to_steps = {col: [] for col in self.feature_names_in_}
        for step_name, step in self.pipeline_.steps:
            if not hasattr(step, "transformers_"):
                continue
            for in_col, transformer in step.transformers_.items():
                col_to_steps[in_col].append((step_name, transformer))
        if kind is not None:
            kind = f"{kind}_transformer"
            col_to_steps = {
                col: steps
                for (col, steps) in col_to_steps.items()
                if steps and steps[-1][0] == kind
            }
        return col_to_steps

    def get_transformers(self, kind=None):
        """Get the final transformer applied to each column.

        Parameters
        ----------
        kind : "numeric", "datetime", "high_cardinality", \
"low_cardinality", "remainder", or None, optional
            Filter results to return only those corresponding to the provided kind.
            None (the default) means return all results.

        Returns
        -------
        dict
            Mapping each column name to the transformer that generated the
            corresponding output columns.
        """
        col_to_steps = self.get_processing_steps(kind=kind)
        transformers = {
            c: steps[-1][1] if steps else None for c, steps in col_to_steps.items()
        }
        return transformers

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


def _get_input_to_outputs_mapping(pipeline_steps):
    mapping = {col: [col] for col in pipeline_steps[0].all_inputs_}
    for step in pipeline_steps:
        for col, outputs_at_previous_step in mapping.items():
            new_outputs = []
            for output in outputs_at_previous_step:
                new_outputs.extend(step.input_to_outputs_.get(output, [output]))
            mapping[col] = new_outputs
    return mapping
