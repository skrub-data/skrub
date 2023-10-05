from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

WORD_TO_ALIAS = {
    "year": "Y",
    "month": "M",
    "day": "D",
    "hour": "H",
    "minute": "min",
    "second": "S",
    "microsecond": "us",
    "nanosecond": "N",
}
TIME_LEVELS = list(WORD_TO_ALIAS)


def is_datetime_parsable(X):
    """Check whether a 1d vector can be converted into a \
    :class:`~pandas.core.indexes.datetimes.DatetimeIndex`.

    Parameters
    ----------
    X : array-like of shape ``(n_sample,)``

    Returns
    -------
    is_dt_parsable : bool
    """
    if len(X.shape) > 1:
        raise ValueError(f"X must be 1d, got shape: {X.shape}.")
    np_dtypes_candidates = [np.object_, np.str_, np.datetime64]
    if any(np.issubdtype(X.dtype, np_dtype) for np_dtype in np_dtypes_candidates):
        try:
            _ = pd.to_datetime(X)
            return True
        except (pd.errors.ParserError, ValueError):
            pass
    return False


def is_date_only(X):
    """Check whether a 1d vector only contains dates.

    Note that ``is_date_only`` being True implies ``is_datetime_parsable`` is True,
    but not the contrary.

    Parameters
    ----------
    X : array-like of shape ``(n_sample,)``

    Returns
    -------
    is_date : bool
    """
    if is_datetime_parsable(X):
        X_t = pd.to_datetime(X)
        return np.all(X_t == X_t.normalize())
    return False


class DatetimeEncoder(TransformerMixin, BaseEstimator):
    """Transforms each datetime column into several numeric columns \
    for temporal features (e.g year, month, day...).

    Constant extracted features are dropped; for instance, if the year is
    always the same in a feature, the extracted "year" column won't be added.
    If the dates are timezone aware, all the features extracted will correspond
    to the provided timezone.

    Parameters
    ----------
    extract_until : {"year", "month", "day", "hour", "minute", "second",
        "microsecond", "nanosecond", None}, default="hour"
        Extract up to this granularity.
        For instance, if you specify "day", only "year", "month", "day" and
        features will be created.
        If ``None``, no feature will be created.

    add_day_of_the_week : bool, default=False
        Add day of the week feature (if day is extracted).
        This is a numerical feature from 0 (Monday) to 6 (Sunday).

    add_total_second : bool, default=True
        Add the total number of seconds since Epoch.

    errors: {"coerce", "raise"}, default="coerce"
        During transform:
        - If ``"coerce"``, then invalid parsing will be set as ``NaT``.
        - If ``"raise"``, then invalid parsing will raise an exception

    Attributes
    ----------
    n_features_out_ : int
        Number of features of the transformed data.

    features_per_column_ : dict[str, list[str]] or dict[int, list[str]]
        Dictionary mapping the column names to the list of features extracted
        for each column.

    format_per_column_ : dict[str, str] or dict[int, str]
        Dictionary mapping the column names to the first non-null example.
        This is how Pandas infer the datetime format.

    See Also
    --------
    GapEncoder :
        Encode dirty categories (strings) by constructing
        latent topics with continuous encoding.

    MinHashEncoder :
        Encode string columns as a numeric array with the minhash method.

    SimilarityEncoder :
        Encode string columns as a numeric array with n-gram string similarity.

    Examples
    --------
    >>> enc = DatetimeEncoder()

    Let's encode the following dates:

    >>> X = [['2022-10-15'], ['2021-12-25'], ['2020-05-18'], ['2019-10-15 12:00:00']]

    >>> enc.fit(X)
    DatetimeEncoder()

    The encoder will output a transformed array
    with four columns ("year", "month", "day" and "hour"):

    >>> enc.transform(X)
    array([[2022.,   10.,   15.,    0.],
           [2021.,   12.,   25.,    0.],
           [2020.,    5.,   18.,    0.],
           [2019.,   10.,   15.,   12.]])
    """

    def __init__(
        self,
        *,
        extract_until="hour",
        add_day_of_the_week=False,
        add_total_second=True,
        errors="coerce",
    ):
        self.extract_until = extract_until
        self.add_day_of_the_week = add_day_of_the_week
        self.add_total_second = add_total_second
        self.errors = errors

    def fit(self, X, y=None):
        """Fit the instance to X.

        In practice, just check keywords and input validity,
        and stores which extracted features are not constant.

        Parameters
        ----------
        X : array-like, shape ``(n_samples, n_features)``
            Data where each column is a datetime feature.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        DatetimeEncoder
            Fitted DatetimeEncoder instance (self).
        """
        if self.extract_until not in TIME_LEVELS and self.extract_until is not None:
            raise ValueError(
                f"'extract_until' options are {TIME_LEVELS}, "
                f"got {self.extract_until!r}."
            )

        errors_options = ["coerce", "raise"]
        if self.errors not in errors_options:
            raise ValueError(
                f"errors options are {errors_options!r}, got {self.errors!r}."
            )

        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)
        X = check_array(X, ensure_2d=True, force_all_finite=False, dtype=None)

        self._select_datetime_cols(X)

        return self

    def _select_datetime_cols(self, X):
        """Select datetime-like columns and infer features to be parsed.

        If the input only contains dates (and no datetimes), only the features
        ["year", "month", "day"] will be filtered with extract_until.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        # Features to extract for each column, after removing constant features
        self.features_per_column_ = defaultdict(list)
        self.format_per_column_ = dict()
        self.n_features_out_ = 0

        if self.extract_until is None:
            levels = []
        else:
            idx_level = TIME_LEVELS.index(self.extract_until)
            levels = TIME_LEVELS[: idx_level + 1]

        columns = getattr(self, "feature_names_in_", list(range(X.shape[1])))
        for col_idx, col in enumerate(columns):
            X_col = X[:, col_idx]

            if is_datetime_parsable(X_col):
                # Pandas use the first non-null item of the array to infer the format.
                X_dt = pd.to_datetime(X_col)
                mask_notnull = X_dt == X_dt
                self.format_per_column_[col] = X_col[mask_notnull][0]

                if is_date_only(X_col):
                    # Keep only date attributes
                    levels = [
                        level for level in levels if level in ["year", "month", "day"]
                    ]

                self.features_per_column_[col] += levels
                self.n_features_out_ += len(levels)

                if self.add_total_second:
                    self.features_per_column_[col].append("total_second")
                    self.n_features_out_ += 1

                if self.add_day_of_the_week:
                    self.features_per_column_[col].append("day_of_week")
                    self.n_features_out_ += 1

    def transform(self, X, y=None):
        """Transform `X` by replacing each datetime column with \
        corresponding numerical features.

        Parameters
        ----------
        X : array-like of shape ``(n_samples, n_features)``
            The data to transform, where each column is a datetime feature.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        X_out : ndarray of shape ``(n_samples, n_features_out_)``
            Transformed input.
        """
        check_is_fitted(self)
        self._check_n_features(X, reset=False)
        self._check_feature_names(X, reset=False)
        X = check_array(X, ensure_2d=True, force_all_finite=False, dtype=None)

        return self._parse_datetime_cols(X)

    def _parse_datetime_cols(self, X):
        """Extract datetime features from the selected columns.

        Parameters
        ----------
        X : ndarray of shape ``(n_samples, n_features)``

        Returns
        -------
        X_out : ndarray of shape ``(n_samples, n_features_out_)``
        """
        columns = getattr(self, "feature_names_in_", list(range(X.shape[1])))
        # X_out must be of dtype float64 to handle np.nan
        X_out = np.empty((X.shape[0], self.n_features_out_), dtype=np.float64)
        offset_idx = 0
        for col_idx, col in enumerate(columns):
            if col in self.features_per_column_:
                # X_j is a DatetimeIndex
                X_col = pd.to_datetime(X[:, col_idx], errors=self.errors)

                features = self.features_per_column_[col]
                for feat_idx, feature in enumerate(features):
                    if feature == "total_second":
                        if X_col.tz is not None:
                            X_col = X_col.tz_convert("utc")
                        # Total seconds since epoch
                        mask_notnull = X_col == X_col
                        X_feature = np.where(
                            mask_notnull,
                            X_col.astype("int64") // 1e9,
                            np.nan,
                        )
                    else:
                        X_feature = getattr(X_col, feature).to_numpy()
                    X_out[:, offset_idx + feat_idx] = X_feature

                offset_idx += len(features)

        return X_out

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Feature names are formatted like: "<column_name>_<new_feature>"
        if the original data has column names, otherwise with format
        "<column_index>_<new_feature>" where `<new_feature>` is one of
        {"year", "month", "day", "hour", "minute", "second",
        "microsecond", "nanosecond", "day_of_week"}.

        Parameters
        ----------
        input_features : None
            Unused, only here for compatibility.

        Returns
        -------
        feature_names : list of str
            List of feature names.
        """
        check_is_fitted(self, "features_per_column_")
        feature_names = []
        for column, features in self.features_per_column_.items():
            feature_names += [f"{column}_{feat}" for feat in features]
        return feature_names

    def _more_tags(self):
        """
        Used internally by sklearn to ease the estimator checks.
        """
        return {
            "X_types": ["2darray", "categorical"],
            "allow_nan": True,
            "_xfail_checks": {"check_dtype_object": "Specific datetime error."},
        }
