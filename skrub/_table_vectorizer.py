import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from . import _selectors as sbs
from . import _utils
from ._check_input import CheckInputDataFrame
from ._clean_null_strings import CleanNullStrings
from ._datetime_encoder import DatetimeEncoder
from ._gap_encoder import GapEncoder
from ._map import Map
from ._to_categorical import ToCategorical
from ._to_datetime import ToDatetime
from ._to_numeric import ToNumeric

HIGH_CARDINALITY_TRANSFORMER = GapEncoder(n_components=30)
LOW_CARDINALITY_TRANSFORMER = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore",
    drop="if_binary",
)
DATETIME_TRANSFORMER = DatetimeEncoder()


def _make_table_vectorizer_pipeline(
    low_cardinality_transformer,
    high_cardinality_transformer,
    numerical_transformer,
    datetime_transformer,
    cardinality_threshold,
    passthrough,
    drop_remainder,
):
    cols = sbs.inv(passthrough)

    cleaning_steps = [
        CheckInputDataFrame(),
        Map(CleanNullStrings(), cols),
        Map(ToDatetime(), cols),
        Map(ToNumeric(), cols),
        Map(ToCategorical(cardinality_threshold), cols),
    ]

    low_card_cat = sbs.categorical() & sbs.cardinality_below(cardinality_threshold)

    feature_extraction_steps = [
        Map(low_cardinality_transformer, low_card_cat - passthrough),
        Map(high_cardinality_transformer, sbs.string() - passthrough),
        Map(numerical_transformer, sbs.numeric() - passthrough),
        Map(datetime_transformer, sbs.anydate() - passthrough),
    ]

    all_steps = cleaning_steps + feature_extraction_steps

    if drop_remainder:
        remainder = cols - sbs.produced_by(*feature_extraction_steps)
        all_steps.append(Map(Drop(), remainder))

    return make_pipeline(*all_steps)


class PassThrough(BaseEstimator):
    __univariate_transformer__ = True

    def fit_transform(self, column):
        return column

    def transform(self, column):
        return column


class Drop(BaseEstimator):
    __univariate_transformer__ = True

    def fit_transform(self, column):
        return []

    def transform(self, column):
        return []


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


class TableVectorizer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        cardinality_threshold=40,
        low_cardinality_transformer=LOW_CARDINALITY_TRANSFORMER,
        high_cardinality_transformer=HIGH_CARDINALITY_TRANSFORMER,
        numerical_transformer="passthrough",
        datetime_transformer=DATETIME_TRANSFORMER,
        passthrough=(),
        drop_remainder=True,
        n_jobs=None,
    ):
        self.cardinality_threshold = cardinality_threshold
        self.low_cardinality_transformer = _utils.clone_if_default(
            low_cardinality_transformer, LOW_CARDINALITY_TRANSFORMER
        )
        self.high_cardinality_transformer = _utils.clone_if_default(
            high_cardinality_transformer, HIGH_CARDINALITY_TRANSFORMER
        )
        self.datetime_transformer = _utils.clone_if_default(
            datetime_transformer, DATETIME_TRANSFORMER
        )
        self.numerical_transformer = numerical_transformer
        self.passthrough = passthrough
        self.drop_remainder = drop_remainder
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

    def fit_transform(self, X, y=None):
        self.pipeline_ = _make_table_vectorizer_pipeline(
            _clone_or_create_transformer(self.low_cardinality_transformer),
            _clone_or_create_transformer(self.high_cardinality_transformer),
            _clone_or_create_transformer(self.numerical_transformer),
            _clone_or_create_transformer(self.datetime_transformer),
            self.cardinality_threshold,
            self.passthrough,
            self.drop_remainder,
        )
        output = self.pipeline_.fit_transform(X)
        self.feature_names_out_ = sbd.column_names(output)
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
