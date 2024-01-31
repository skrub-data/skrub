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
):
    if hasattr(passthrough, "__iter__"):
        passthrough = list(passthrough)
    if isinstance(passthrough, list) and not passthrough:
        # get a shorter display in scikit-learn _html_repr_ by using the default value
        cols = sbs.all()
    else:
        cols = sbs.inv(passthrough)

    cleaning_steps = [
        ("check_input", CheckInputDataFrame()),
        ("convert_dtypes", Map(PandasConvertDTypes(), cols)),
        ("clean_null_strings", Map(CleanNullStrings(), cols)),
        ("to_datetime", Map(ToDatetime(), cols)),
        ("to_numeric", Map(ToNumeric(), cols)),
        ("to_categorical", Map(ToCategorical(cardinality_threshold - 1), cols)),
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
        feature_extraction_steps.append(Map(transformer, selector))
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


# auto_wrap_output_keys = () is so that the TransformerMixin does not wrap
# transform or provide set output (we always produce dataframes of the correct
# type with the correct columns and we don't want the wrapper.) other ways to
# disable it would be not inheriting from TransformerMixin, not defining
# get_feature_names_out


class TableVectorizer(TransformerMixin, BaseEstimator, auto_wrap_output_keys=()):
    """Transform a dataframe to a numerical array.

    TODO
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
        return _make_table_vectorizer_pipeline(
            _clone_or_create_transformer(self.low_cardinality_transformer),
            _clone_or_create_transformer(self.high_cardinality_transformer),
            _clone_or_create_transformer(self.numeric_transformer),
            _clone_or_create_transformer(self.datetime_transformer),
            _clone_or_create_transformer(self.remainder_transformer),
            self.cardinality_threshold,
            self.passthrough,
        )

    def fit_transform(self, X, y=None):
        self.pipeline_ = self.make_pipeline()
        output = self.pipeline_.fit_transform(X)
        self.feature_names_in_ = self.pipeline_.named_steps[
            "check_input"
        ].feature_names_in_
        self.feature_names_out_ = sbd.column_names(output)
        self.input_to_outputs_ = _get_input_to_outputs_mapping(
            list(self.pipeline_.named_steps.values())[1:]
        )
        return output

    def transform(self, X):
        return self.pipeline_.transform(X)

    def get_feature_names_out(self):
        check_is_fitted(self, "feature_names_out_")
        return np.asarray(self.feature_names_out_)

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
        col_to_steps = self.get_processing_steps(kind=kind)
        transformers = {
            c: steps[-1][1] if steps else None for c, steps in col_to_steps.items()
        }
        return transformers


def _get_input_to_outputs_mapping(pipeline_steps):
    mapping = {col: [col] for col in pipeline_steps[0].all_inputs_}
    for step in pipeline_steps:
        for col, outputs_at_previous_step in mapping.items():
            new_outputs = []
            for output in outputs_at_previous_step:
                new_outputs.extend(step.input_to_outputs_.get(output, [output]))
            mapping[col] = new_outputs
    return mapping
