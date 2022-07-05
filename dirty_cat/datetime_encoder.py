import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List

from sklearn.utils.validation import check_is_fitted

from dirty_cat.utils import check_input

# Some functions need aliases
WORD_TO_ALIAS = {"year": "Y", "month": "M", "day": "D", "hour": "H", "minute": "min", "second": "S",
                 "millisecond": "ms", "microsecond": "us", "nanosecond": "N"}
TIME_LEVELS = ["year", "month", "day", "hour", "minute", "second", "millisecond", "microsecond", "nanosecond"]


class DatetimeEncoder(BaseEstimator, TransformerMixin):
    """
    This encoder transforms each datetime column into several numeric columns corresponding to temporal features,
    e.g year, month, day...
    Constant extracted features are dropped ; for instance, if the year is always the same in a feature, the extracted "year" column won't be added.
    If the dates are timezone aware, all the features extracted will correspond to the provided timezone.

    Parameters
    ----------
    extract_until : {"year", "month", "day", "hour", "minute", "second", "millisecond", "microsecond", "nanosecond"}, default="hour"
        Extract up to this granularity. If all features have not been extracted, add the "total_time" feature, which contains
        the time to epoch (in seconds).
        For instance, if you specify "day", only "year", "month", "day" and "total_time" features will be created.
    add_day_of_the_week: bool, default=False
        Add day of the week feature (if day is extracted). This is a numerical feature from 0 (Monday) to 6 (Sunday).

    Attributes
    ----------
    n_features_out_: int
        Number of features of the transformed data.
    features_per_column_: Dict[int, List[str]]
        Dictionary mapping the index of the original columns
        to the list of features extracted for each column.
    col_names_: List[str]
        List of the names of the features of the input data, if input data was a pandas DataFrame, otherwise None.
    """

    def __init__(self,
                 extract_until="hour",
                 add_day_of_the_week=False):
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
                f'"extract_until" should be one of {TIME_LEVELS}, got {self.extract_until}.'
            )

    def _extract_from_date(self, date_series, feature):
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
                return (pd.to_datetime(date_series) - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
            else:
                return (pd.DatetimeIndex(date_series).tz_convert("utc") - pd.Timestamp("1970-01-01", tz="utc")) // pd.Timedelta('1s')

    def fit(self, X, y=None):
        """
        Fit the DatetimeEncoder to X. In practice, just stores which extracted features
        are not constant.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data where each column is a datetime feature.

        Returns
        -------
        self
            Fitted DatetimeEncoder instance.
        """
        self._validate_keywords()
        # Columns to extract for each column, before taking into account constant columns
        self._to_extract = TIME_LEVELS[:TIME_LEVELS.index(self.extract_until) + 1]
        if self.add_day_of_the_week:
            self._to_extract.append("dayofweek")
        if isinstance(X, pd.DataFrame):
            self.col_names_ = X.columns
        else:
            self.col_names_ = None
        X = check_input(X)
        self.features_per_column_ = {}  # Features to extract for each column, after removing constant features
        for i in range(X.shape[1]):
            self.features_per_column_[i] = []
        # Check which columns are constant
        for i in range(X.shape[1]):
            for feature in self._to_extract:
                if np.nanstd(self._extract_from_date(X[:, i], feature)) > 0:
                    self.features_per_column_[i].append(feature)
            # If some date features have not been extracted, then add the "total_time" feature, which contains the full
            # time to epoch
            remainder = (pd.to_datetime(X[:, i]) - pd.to_datetime(pd.DatetimeIndex(X[:, i]).floor(
                WORD_TO_ALIAS[self.extract_until]))).seconds.to_numpy()
            if np.nanstd(remainder) > 0:
                self.features_per_column_[i].append("total_time")

        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = len(np.concatenate(list(self.features_per_column_.values())))

        return self

    def transform(self, X, y=None):
        """ Transform X by replacing each datetime column with corresponding numerical features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to transform, where each column is a datetime feature.

        Returns
        -------
        array, shape (n_samples, n_features_out_)
            Transformed input.
        """
        check_is_fitted(self, attributes=["n_features_in_", "n_features_out_", "features_per_column_"])
        X = check_input(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Number of features in the input data ({X.shape[1]}) does not match the number of features "
                f"seen during fit ({self.n_features_in_})."
            )
        # Create a new array with the extracted features, choosing only features that weren't constant during fit
        X_ = np.empty((X.shape[0], self.n_features_out_), dtype=np.float64)
        idx = 0
        for i in range(X.shape[1]):
            for j, feature in enumerate(self.features_per_column_[i]):
                X_[:, idx + j] = self._extract_from_date(X[:, i], feature)
            idx += len(self.features_per_column_[i])
        return X_

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Returns clean feature names with format "<column_name>_<new_feature>"
        if the original data has column names, otherwise with format
        "<column_index>_<new_feature>". new_feature is one of ["year", "month",
        "day", "hour", "minute", "second", "millisecond", "microsecond",
        "nanosecond", "dayofweek"]
        """
        feature_names = []
        for i in self.features_per_column_.keys():
            prefix = str(i) if self.col_names_ is None else self.col_names_[i]
            for feature in self.features_per_column_[i]:
                feature_names.append(f"{prefix}_{feature}")
        return feature_names

    def get_feature_names(self, input_features=None) -> List[str]:
        """
        Ensures compatibility with sklearn < 1.0, and returns the output of
        get_feature_names_out.
        """
        return self.get_feature_names_out()

