import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted

from . import _selectors, _utils
from ._check_input import CheckInputDataFrame
from ._clean_null_strings import CleanNullStrings
from ._dataframe import asdfapi
from ._datetime_encoder import DatetimeEncoder
from ._gap_encoder import GapEncoder
from ._map_cols import MapCols
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
    cardinality_threshold,
    low_cardinality_transformer,
    high_cardinality_transformer,
    numerical_transformer,
    datetime_transformer,
):
    return make_pipeline(
        CheckInputDataFrame(),
        MapCols(CleanNullStrings()),
        MapCols(ToDatetime()),
        MapCols(ToNumeric()),
        MapCols(ToCategorical(cardinality_threshold)),
        MapCols(low_cardinality_transformer, cols=_selectors.categorical()),
        MapCols(high_cardinality_transformer, cols=_selectors.string()),
        MapCols(numerical_transformer, cols=_selectors.numeric()),
        MapCols(datetime_transformer, cols=_selectors.anydate()),
    )


class PassThrough(BaseEstimator):
    __univariate_transformer__ = True

    def fit_transform(self, column):
        return NotImplemented


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
            self.cardinality_threshold,
            _clone_or_create_transformer(self.low_cardinality_transformer),
            _clone_or_create_transformer(self.high_cardinality_transformer),
            _clone_or_create_transformer(self.numerical_transformer),
            _clone_or_create_transformer(self.datetime_transformer),
        )
        output = self.pipeline_.fit_transform(X)
        self.feature_names_out_ = asdfapi(output).column_names
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
