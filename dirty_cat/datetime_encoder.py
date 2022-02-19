from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from dirty_cat.utils import check_input
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar



class DatetimeEncoder(TransformerMixin, BaseEstimator):
    """
    This encoder transforms each datetime column into several numeric columns corresponding to temporal features,
    e.g year, month, day...
    If the dates are timezone aware, all the features extracted will correspond to the provided timezone.
    Parameters
    ----------
    extract_until : {"year", "month", "day", "hour", "minute", "second",
    "millisecond", "microsecond", "nanosecond"}, default="hour"
        Extract up to this granularity. For instance, if you specify "day", only "year", "month", and "day"
        features will be created. The rest will me gathered into the "other" feature.
    add_day_of_the_week: bool, default=True
        Add day of the week feature (if day is extracted).
    add_holidays : bool, default=False
        Whether to add a categorical variable encoding if the day of the date is a holiday (and if day is extracted).
        Uses pandas calendar, which for now only supports US holidays.
    """
    def __init__(self,
                 extract_until="hour",
                 add_day_of_the_week=True,
                 add_holidays=False):
        to_extract = ["year", "month", "day", "hour", "minute", "second", "millisecond", "microsecond", "nanosecond"]
        self.extract_until = extract_until
        self._validate_keywords()
        self.to_extract_full = to_extract[:to_extract.index(extract_until) + 1]
        self.to_extract_full.append("other")
        self.add_day_of_the_week = add_day_of_the_week
        self.add_holidays = add_holidays
        # number of new columns created per feature, before removing constant columns
        self.n_features_per_col_full = len(self.to_extract_full) + self.add_day_of_the_week + self.add_holidays
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
        X = check_input(X)
        self.to_extract = {}  # Features to extract for each column, after removing constant features
        for i in range(X.shape[1]):
            self.to_extract[i] = []
        # Check which columns are constant
        for i in range(X.shape[1]):
            for feature in self.to_extract_full:
                print(feature)
                print(self._extract_from_date(X[:, i], feature))
                print(self._extract_from_date(X[:, i], feature).dtype)
                if np.nanstd(self._extract_from_date(X[:, i], feature)) > 0:
                    self.to_extract[i].append(feature)
            if "day" in self.to_extract[i]:
                if self.add_day_of_the_week:
                    self.to_extract[i].append("dayofweek")
                if self.add_holidays:
                    self.to_extract[i].append("holiday")

        self.n_features_out = len(np.concatenate(list(self.to_extract.values())))

        return self

    def transform(self, X, y=None):
        X = check_input(X)
        # Create a new dataframe with the extracted features, choosing only features that weren't constant during fit
        X_ = np.empty((X.shape[0], self.n_features_out), dtype=np.float64)
        idx = 0
        for i in range(X.shape[1]):
            for j, feature in enumerate(self.to_extract[i]):
                X_[:, idx + j] = self._extract_from_date(X[:, i], feature)
            idx += len(self.to_extract[i])
        return X_