from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from skrub._utils import check_input

# Required for ignoring lines too long in the docstrings
# flake8: noqa: E501


WORD_TO_ALIAS: dict[str, str] = {
    "year": "Y",
    "month": "M",
    "day": "D",
    "hour": "H",
    "minute": "min",
    "second": "S",
    "millisecond": "ms",
    "microsecond": "us",
    "nanosecond": "N",
}
TIME_LEVELS: list[str] = list(WORD_TO_ALIAS.keys())
AcceptedTimeValues = Literal[
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
]


class DatetimeEncoder(BaseEstimator, TransformerMixin):
    """Transforms each datetime column into several numeric columns for temporal features (e.g year, month, day...).

    Constant extracted features are dropped; for instance, if the year is
    always the same in a feature, the extracted "year" column won't be added.
    If the dates are timezone aware, all the features extracted will correspond
    to the provided timezone.

    Parameters
    ----------
    extract_until : {'year', 'month', 'day', 'hour', 'minute', 'second', 'millisecond', 'microsecond', 'nanosecond'}, default='hour'
        Extract up to this granularity.
        If all features have not been extracted, add the 'total_time' feature,
        which contains the time to epoch (in seconds).
        For instance, if you specify 'day', only 'year', 'month', 'day' and
        'total_time' features will be created.
    add_day_of_the_week : bool, default=False
        Add day of the week feature (if day is extracted).
        This is a numerical feature from 0 (Monday) to 6 (Sunday).

    Attributes
    ----------
    n_features_in_ : int
        Number of features in the data seen during fit.
    n_features_out_ : int
        Number of features of the transformed data.
    features_per_column_ : mapping of int to list of str
        Dictionary mapping the index of the original columns
        to the list of features extracted for each column.
    col_names_ : None or list of str
        List of the names of the features of the input data,
        if input data was a pandas DataFrame, otherwise None.

    See Also
    --------
    :class:`skrub.GapEncoder` :
        Encodes dirty categories (strings) by constructing latent topics with continuous encoding.
    :class:`skrub.MinHashEncoder` :
        Encode string columns as a numeric array with the minhash method.
    :class:`skrub.SimilarityEncoder` :
        Encode string columns as a numeric array with n-gram string similarity.

    Examples
    --------
    >>> enc = DatetimeEncoder()

    Let's encode the following dates:

    >>> X = [['2022-10-15'], ['2021-12-25'], ['2020-05-18'], ['2019-10-15 12:00:00']]

    >>> enc.fit(X)
    DatetimeEncoder()

    The encoder will output a transformed array
    with four columns (year, month, day and hour):

    >>> enc.transform(X)
    array([[2022.,   10.,   15.,    0.],
           [2021.,   12.,   25.,    0.],
           [2020.,    5.,   18.,    0.],
           [2019.,   10.,   15.,   12.]])
    """

    n_features_in_: int
    n_features_out_: int
    features_per_column_: dict[int, list[str]]
    col_names_: list[str] | None

    def __init__(
        self,
        *,
        extract_until: AcceptedTimeValues = "hour",
        add_day_of_the_week: bool = False,
    ):
        self.extract_until = extract_until
        self.add_day_of_the_week = add_day_of_the_week

    def _more_tags(self):
        """
        Used internally by sklearn to ease the estimator checks.
        """
        return {"X_types": ["categorical"]}

    def _validate_keywords(self):
        if self.extract_until not in TIME_LEVELS:
            raise ValueError(
                f'"extract_until" should be one of {TIME_LEVELS}, '
                f"got {self.extract_until}. "
            )

    @staticmethod
    def _extract_from_date(date_series: pd.Series, feature: str):
        if feature == "year":
            return pd.DatetimeIndex(date_series).year.to_numpy()
        elif feature == "month":
            return pd.DatetimeIndex(date_series).month.to_numpy()
        elif feature == "day":
            return pd.DatetimeIndex(date_series).day.to_numpy()
        elif feature == "hour":
            return pd.DatetimeIndex(date_series).hour.to_numpy()
        elif feature == "minute":
            return pd.DatetimeIndex(date_series).minute.to_numpy()
        elif feature == "second":
            return pd.DatetimeIndex(date_series).second.to_numpy()
        elif feature == "millisecond":
            return pd.DatetimeIndex(date_series).millisecond.to_numpy()
        elif feature == "microsecond":
            return pd.DatetimeIndex(date_series).microsecond.to_numpy()
        elif feature == "nanosecond":
            return pd.DatetimeIndex(date_series).nanosecond.to_numpy()
        elif feature == "dayofweek":
            return pd.DatetimeIndex(date_series).dayofweek.to_numpy()
        elif feature == "total_time":
            tz = pd.DatetimeIndex(date_series).tz
            # Compute the time in seconds from the epoch time UTC
            if tz is None:
                return (
                    pd.to_datetime(date_series) - pd.Timestamp("1970-01-01")
                ) // pd.Timedelta("1s")
            else:
                return (
                    pd.DatetimeIndex(date_series).tz_convert("utc")
                    - pd.Timestamp("1970-01-01", tz="utc")
                ) // pd.Timedelta("1s")

    def fit(self, X, y=None) -> "DatetimeEncoder":
        """Fit the instance to X.

        In practice, just stores which extracted features are not constant.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data where each column is a datetime feature.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        :obj:`DatetimeEncoder`
            Fitted :class:`DatetimeEncoder` instance (self).
        """
        self._validate_keywords()
        # Columns to extract for each column,
        # before taking into account constant columns
        self._to_extract = TIME_LEVELS[: TIME_LEVELS.index(self.extract_until) + 1]
        if self.add_day_of_the_week:
            self._to_extract.append("dayofweek")
        if isinstance(X, pd.DataFrame):
            self.col_names_ = X.columns.to_list()
        else:
            self.col_names_ = None
        X = check_input(X)
        # Features to extract for each column, after removing constant features
        self.features_per_column_ = {}
        for i in range(X.shape[1]):
            self.features_per_column_[i] = []
        # Check which columns are constant
        for i in range(X.shape[1]):
            for feature in self._to_extract:
                if np.nanstd(self._extract_from_date(X[:, i], feature)) > 0:
                    self.features_per_column_[i].append(feature)
            # If some date features have not been extracted, then add the
            # "total_time" feature, which contains the full time to epoch
            remainder = (
                pd.to_datetime(X[:, i])
                - pd.to_datetime(
                    pd.DatetimeIndex(X[:, i]).floor(WORD_TO_ALIAS[self.extract_until])
                )
            ).seconds.to_numpy()
            if np.nanstd(remainder) > 0:
                self.features_per_column_[i].append("total_time")

        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = len(
            np.concatenate(list(self.features_per_column_.values()))
        )

        return self

    def transform(self, X, y=None) -> np.ndarray:
        """Transform `X` by replacing each datetime column with corresponding numerical features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to transform, where each column is a datetime feature.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        :obj:`~numpy.ndarray`, shape (n_samples, `n_features_out_`)
            Transformed input.
        """
        check_is_fitted(
            self,
            attributes=["n_features_in_", "n_features_out_", "features_per_column_"],
        )
        X = check_input(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"The number of features in the input data ({X.shape[1]}) "
                "does not match the number of features "
                f"seen during fit ({self.n_features_in_}). "
            )
        # Create a new array with the extracted features,
        # choosing only features that weren't constant during fit
        X_ = np.empty((X.shape[0], self.n_features_out_), dtype=np.float64)
        idx = 0
        for i in range(X.shape[1]):
            for j, feature in enumerate(self.features_per_column_[i]):
                X_[:, idx + j] = self._extract_from_date(X[:, i], feature)
            idx += len(self.features_per_column_[i])
        return X_

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Return clean feature names.

        Feature names are formatted like: "<column_name>_<new_feature>"
        if the original data has column names, otherwise with format
        "<column_index>_<new_feature>" where `<new_feature>` is one of
        {"year", "month", "day", "hour", "minute", "second", "millisecond",
        "microsecond", "nanosecond", "dayofweek"}.

        Parameters
        ----------
        input_features : None
            Unused, only here for compatibility.

        Returns
        -------
        list of str
            List of feature names.
        """
        feature_names = []
        for i in self.features_per_column_.keys():
            prefix = str(i) if self.col_names_ is None else self.col_names_[i]
            for feature in self.features_per_column_[i]:
                feature_names.append(f"{prefix}_{feature}")
        return feature_names
