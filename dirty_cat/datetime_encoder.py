from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class DatetimeEncoder(TransformerMixin, BaseEstimator):
    def __init__(self,
                 stop_extraction="minute",
                 add_day_of_the_week=True,
                 add_holidays=False):
        # TODO validate_keywords
        # TODO doc
        """
        Parameters
        ----------
        stop_extraction : {"year", "month", "day", "hour", "minute", "second",
        "millisecond", "microsecond", "nanosecond"}, default="minutes"
            Extract up to this granularity. For instance, if you specify "days", only "year", "month", and "day"
            features will be created.
        add_day_of_the_week: bool, default=True
            Add day of the week feature (if day is extracted).
        add_holidays : bool, default=False
            Whether to add a categorical variable encoding if the day of the date is a holiday. For now uses pandas
            calendar, which only support US holidays.
        """
        to_extract = ["year", "month", "day", "hour", "minute", "second", "millisecond", "microsecond", "nanosecond"]
        self.to_extract_full = to_extract[:to_extract.index(stop_extraction) + 1]
        self.add_day_of_the_week = add_day_of_the_week
        self.add_holidays = add_holidays
        # number of new columns created per feature, before removing constant columns
        self.n_features_per_col_full = len(self.to_extract_full) + self.add_day_of_the_week + self.add_holidays

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
            return pd.DatetimeIndex(date_series).microsecond.to_numpy()
        elif feature == "microsecond":
            return pd.DatetimeIndex(date_series).microsecond.to_numpy()
        elif feature == "nanosecond":
            return pd.DatetimeIndex(date_series).nanosecond.to_numpy()
        elif feature == "dayofweek":
            return pd.DatetimeIndex(date_series).dayofweek.to_numpy()
        elif feature == "holiday":
            return None

    def _create_datetime_features(self, X):
        """Create datetime features
        Parameters
        ----------
        X : array-like of shape (n_samples, 1)
            The input samples, containing 1 datetime column.
        Returns
        -------
        X_new : array-like of shape (n_samples, n_features_out)
            The transformed dataset.
        """
        X_ = np.empty((X.shape[0], self.n_features_per_col_full), dtype=np.int64)
        for i, feature in enumerate(self.to_extract):
            X_[:, i] = self._extract_from_date(X, feature)
        return X_

    def fit(self, X, y=None):
        self.to_extract = {}  # Features to extract for each column, after removing constant features
        for i in range(X.shape[1]):
            self.to_extract[i] = []
        # Check which columns are constant
        for i in range(X.shape[1]):
            for feature in self.to_extract_full:
                if self._extract_from_date(X[:, i], feature).std() > 0:
                    self.to_extract[i].append(feature)
            if "day" in self.to_extract[i]:
                if self.add_day_of_the_week:
                    self.to_extract[i].append("dayofweek")
                if self.add_holidays:
                    self.to_extract[i].append("holiday")

        self.n_features_out = len(np.concatenate(list(self.to_extract.values())))

    def transform(self, X, y=None):
        X_ = np.empty((X.shape[0], self.n_features_out), dtype=np.int64)
        idx = 0
        for i in range(X.shape[1]):
            for j, feature in enumerate(self.to_extract[i]):
                X_[:, idx + j] = self._extract_from_date(X[:, i], feature)
            idx += len(self.to_extract[i])
        return X_