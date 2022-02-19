from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from dirty_cat.utils import check_input
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from typing import Union, Optional, List



class DatetimeEncoder(TransformerMixin, BaseEstimator):
    """
    This encoder transforms each datetime column into several numeric columns corresponding to temporal features,
    e.g year, month, day...
    If one of the feature is constant, this feature is dropped.
    If the dates are timezone aware, all the features extracted will correspond to the provided timezone.

    Parameters
    ----------
    extract_until : {"year", "month", "day", "hour", "minute", "second",
    "millisecond", "microsecond", "nanosecond"}, default="hour"
        Extract up to this granularity, and gather the rest into the "other" feature.
        For instance, if you specify "day", only "year", "month", "day" and "other" features will be created.
        The "other" feature will be a numerical value expressed in the "extract_until" unit.
    add_day_of_the_week: bool, default=False
        Add day of the week feature (if day is extracted). This is a numerical feature from 0 to 6.
    add_holidays : bool, default=False
        Whether to add a numerical variable encoding if the day of the date is a holiday (1 for holiday,
        0 for non-holiday).
        Uses pandas calendar, which for now only contains US holidays.

    Attributes
    ----------
    n_features_out: int
        Number of features of the transformed data.
    features_per_column: Dict[int, List[str]]
        A dictionary mapping the index of the original columns
        to the list of features extracted for each column.
    """
    def __init__(self,
                 extract_until="hour",
                 add_day_of_the_week=False,
                 add_holidays=False):
        to_extract = ["year", "month", "day", "hour", "minute", "second", "millisecond", "microsecond", "nanosecond"]
        self.extract_until = extract_until
        self._validate_keywords()
        self.to_extract_full = to_extract[:to_extract.index(extract_until) + 1]
        self.to_extract_full.append("other")
        self.add_day_of_the_week = add_day_of_the_week
        self.add_holidays = add_holidays
        # Some functions need aliases
        self.word_to_alias = {"year": "Y", "month": "M", "day": "D", "hour": "H", "minute": "min", "second": "S",
                              "millisecond": "ms", "microsecond": "us", "nanosecond": "N"}

    def _validate_keywords(self):
        if self.extract_until not in ["year", "month", "day", "hour", "minute", "second", "millisecond", "microsecond", "nanosecond"]:
            msg = (
                """extract_until should be one of ["year", "month", "day", "hour", "minute", "second", "millisecond", 
                "microsecond", "nanosecond"], got {0}.""".format(
                    self.extract_until
                )
            )
            raise ValueError(msg)

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
            # Create an indicator for holidays
            cal = calendar()
            holidays = cal.holidays(start=date_series.min(), end=date_series.max())
            return np.isin(date_series, holidays).astype(int)
        elif feature == "other":
            # Gather all the variables below the extract_until into one numerical variable
            res = (pd.to_datetime(date_series) - pd.to_datetime(pd.DatetimeIndex(date_series).floor(
                self.word_to_alias[self.extract_until]))).to_numpy()
            # Convert to the extract_until unit (e.g if I extract until "minute", then convert to minutes)
            return res / pd.to_timedelta(1, self.word_to_alias[self.extract_until])

    def fit(self, X, y=None):
        """
        Fit the DatetimeEncoder to X. In practice, just stores which extracted features
        are not constant.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            Data where each column is a datetime feature.
        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            self.colnames = X.columns
        else:
            self.colnames = None
        X = check_input(X)
        self.features_per_column = {}  # Features to extract for each column, after removing constant features
        for i in range(X.shape[1]):
            self.features_per_column[i] = []
        # Check which columns are constant
        for i in range(X.shape[1]):
            for feature in self.to_extract_full:
                if np.nanstd(self._extract_from_date(X[:, i], feature)) > 0:
                    self.features_per_column[i].append(feature)
            if self.add_day_of_the_week:
                self.features_per_column[i].append("dayofweek")
            if self.add_holidays:
                self.features_per_column[i].append("holiday")

        self.n_features_out = len(np.concatenate(list(self.features_per_column.values())))

        return self

    def transform(self, X, y=None):
        """ Transform X by replacing each column with corresponding numerical features.
        Parameters
        ----------
        X : array-like, shape (n_samples, ) or (n_samples, 1)
            The data to transform, where each column is a datetime feature.
        Returns
        -------
        array, shape (n_samples, n_features_out)
            Transformed input.
        """
        X = check_input(X)
        # Create a new dataframe with the extracted features, choosing only features that weren't constant during fit
        X_ = np.empty((X.shape[0], self.n_features_out), dtype=np.float64)
        idx = 0
        for i in range(X.shape[1]):
            for j, feature in enumerate(self.features_per_column[i]):
                X_[:, idx + j] = self._extract_from_date(X[:, i], feature)
            idx += len(self.features_per_column[i])
        return X_

    def get_feature_names(self) -> List[str]:
        """
        Returns clean feature names with format "<column_name>_<new_feature>"
        if the original data has column names, otherwise with format
        "<column_indice>_<new_feature>". new_feature is one of ["year", "month",
        "day", "hour", "minute", "second", "millisecond", "microsecond",
        "nanosecond", "dayofweek", "holiday"]
        """
        feature_names = []
        for i in self.features_per_column.keys():
            prefix = str(i) if self.colnames is None else self.colnames[i]
            for feature in self.features_per_column[i]:
                feature_names.append(prefix + "_" + feature)
        return feature_names

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Ensures compatibility with sklearn >= 1.0, and returns the output of
        get_feature_names.
        """
        return self.get_feature_names()