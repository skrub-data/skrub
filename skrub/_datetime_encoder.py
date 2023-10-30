from __future__ import annotations

from typing import Literal, TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from skrub._utils import check_input
if TYPE_CHECKING:
    from dataframe_api import Column

WORD_TO_ALIAS: dict[str, str] = {
    "year": "Y",
    "month": "M",
    "day": "D",
    "hour": "H",
    "minute": "min",
    "second": "S",
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
    "microsecond",
    "nanosecond",
]


class DatetimeEncoder(BaseEstimator, TransformerMixin):
    """Transform each datetime column into several numeric columns \
    for temporal features (e.g. "year", "month", "day"...).

    Constant extracted features are dropped; for instance, if the year is
    always the same in a feature, the extracted "year" column won't be added.
    If the dates are timezone aware, all the features extracted will correspond
    to the provided timezone.

    Parameters
    ----------
    extract_until : {"year", "month", "day", "hour", "minute", "second",
        "microsecond", "nanosecond", None}, default="hour"
        Extract up to this granularity.
        If all non-constant features have not been extracted,
        add the "total_time" feature, which contains the time to epoch (in seconds).
        For instance, if you specify "day", only "year", "month", "day" and
        "total_time" features will be created.
        If None, only the "total_time" feature will be created.
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

    n_features_in_: int
    n_features_out_: int
    features_per_column_: dict[int, list[str]]
    col_names_: list[str] | None

    def __init__(
        self,
        *,
        extract_until: AcceptedTimeValues | None = "hour",
        add_day_of_the_week: bool = False,
    ):
        self.extract_until = extract_until
        self.add_day_of_the_week = add_day_of_the_week

    def _more_tags(self):
        """
        Used internally by sklearn to ease the estimator checks.
        """
        return {
            "X_types": ["2darray", "categorical"],
            "allow_nan": True,
            "_xfail_checks": {"check_dtype_object": "Specific datetime error."},
        }

    def _validate_keywords(self):
        if self.extract_until not in TIME_LEVELS and self.extract_until is not None:
            raise ValueError(
                f'"extract_until" should be one of {TIME_LEVELS}, '
                f"got {self.extract_until}. "
            )

    @staticmethod
    def _extract_from_date(date_series: Column, feature: str):
        if feature == "year":
            return date_series.year()
        elif feature == "month":
            return date_series.month()
        elif feature == "day":
            return date_series.day()
        elif feature == "hour":
            return date_series.hour()
        elif feature == "minute":
            return date_series.minute()
        elif feature == "second":
            return date_series.second()
        elif feature == "microsecond":
            return date_series.microsecond()
        elif feature == "nanosecond":
            if hasattr(date_series, 'nanosecond'):
                return date_series.nanosecond()
            else:
                raise AttributeError(
                    f"`nanosecond` is not part of the DataFrame API and so support is not guaranteed across all libraries. "
                    "In particular, it is not supported for {date_series.__class__.__name__}"
                )
        elif feature == "dayofweek":
            return date_series.iso_weekday() - 1
        elif feature == "total_time":
            # Compute the time in seconds from the epoch time UTC
            return date_series.unix_timestamp()  # type: ignore

    def fit(self, X: ArrayLike, y=None) -> "DatetimeEncoder":
        """Fit the instance to ``X``.

        In practice, just check keywords and input validity,
        and stores which extracted features are not constant.

        Parameters
        ----------
        X : array-like, shape (``n_samples``, ``n_features``)
            Data where each column is a datetime feature.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        DatetimeEncoder
            Fitted DatetimeEncoder instance (self).
        """
        self._validate_keywords()
        if not hasattr(X, "__dataframe_consortium_standard__"):
            X = check_input(X)
            X = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
        X = X.__dataframe_consortium_standard__().collect()
        n_colums = len(X.column_names)
        self.col_names_ = X.column_names
        # Features to extract for each column, after removing constant features
        self.features_per_column_ = {}
        for i in range(n_colums):
            self.features_per_column_[i] = []
        # Check which columns are constant
        for i in range(n_colums):
            column = X.col(X.column_names[i])
            if self.extract_until is None:
                if float(self._extract_from_date(column, "total_time").std()) > 0:
                    self.features_per_column_[i].append("total_time")
            else:
                for feature in TIME_LEVELS:
                    if float(self._extract_from_date(column, feature).std()) > 0:
                        if TIME_LEVELS.index(feature) <= TIME_LEVELS.index(
                            self.extract_until
                        ):
                            self.features_per_column_[i].append(feature)
                        # we add a total_time feature, which contains the full
                        # time to epoch, if there is at least one
                        # feature that has not been extracted and is not constant
                        if TIME_LEVELS.index(feature) > TIME_LEVELS.index(
                            self.extract_until
                        ):
                            self.features_per_column_[i].append("total_time")
                            break
                # Add day of the week feature if needed
                if (
                    self.add_day_of_the_week
                    and float(self._extract_from_date(column, "dayofweek").std()) > 0
                ):
                    self.features_per_column_[i].append("dayofweek")

        self.n_features_in_ = n_colums
        self.n_features_out_ = len(
            np.concatenate(list(self.features_per_column_.values()))
        )

        return self

    def transform(self, X: ArrayLike, y=None) -> NDArray:
        """Transform ``X`` by replacing each datetime column with \
        corresponding numerical features.

        Parameters
        ----------
        X : array-like, shape (``n_samples``, ``n_features``)
            The data to transform, where each column is a datetime feature.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        ndarray, shape (``n_samples``, ``n_features_out_``)
            Transformed input.
        """
        if not hasattr(X, "__dataframe_consortium_standard__"):
            X = check_input(X)
            X = pd.DataFrame(X, columns=[str(i) for i in range(X.shape[1])])
        X = X.__dataframe_consortium_standard__()
        n_columns = len(X.column_names)
        check_is_fitted(
            self,
            attributes=["n_features_in_", "n_features_out_", "features_per_column_"],
        )
        # X = check_input(X)
        if n_columns != self.n_features_in_:
            raise ValueError(
                f"The number of features in the input data ({n_columns}) "
                "does not match the number of features "
                f"seen during fit ({self.n_features_in_}). "
            )
        # Create a new array with the extracted features,
        # choosing only features that weren't constant during fit
        # X_ = np.empty((X.shape()[0], self.n_features_out_), dtype=np.float64)
        features_to_select = []
        idx = 0
        for i in range(n_columns):
            column = X.col(X.column_names[i])
            for j, feature in enumerate(self.features_per_column_[i]):
                features_to_select.append(
                    self._extract_from_date(column, feature).rename(f"{feature}_{i}")
                )
            idx += len(self.features_per_column_[i])
        X = X.assign(*features_to_select).select(*(feature.name for feature in features_to_select))
        return X.collect().to_array("float64")

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Return clean feature names.

        Feature names are formatted like: "<column_name>_<new_feature>"
        if the original data has column names, otherwise with format
        "<column_index>_<new_feature>" where `<new_feature>` is one of
        {"year", "month", "day", "hour", "minute", "second",
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
