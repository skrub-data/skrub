from datetime import UTC, datetime

try:
    import pandas as pd
except ImportError:
    pass
try:
    import polars as pl
except ImportError:
    pass

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline

from . import _dataframe as sbd
from ._map import Map
from ._to_datetime import ToDatetime

_TIME_LEVELS = [
    "year",
    "month",
    "day",
    "hour",
    "minute",
    "second",
    "microsecond",
    "nanosecond",
]


@sbd.dispatch
def _is_date(column):
    raise NotImplementedError()


@_is_date.specialize("pandas")
def _is_date_pandas(column):
    return (column.dt.date == column).all()


@_is_date.specialize("polars")
def _is_date_polars(column):
    return (column.dt.date() == column).all()


@sbd.dispatch
def _get_dt_feature(column, feature):
    raise NotImplementedError()


@_get_dt_feature.specialize("pandas")
def _get_dt_feature_pandas(column, feature):
    if feature == "total_seconds":
        if column.dt.tz is None:
            epoch = datetime(1970, 1, 1)
        else:
            epoch = datetime(1970, 1, 1, tzinfo=UTC)
        return ((column - epoch) / pd.Timedelta("1s")).astype("float32")
    assert feature in _TIME_LEVELS + ["day_of_the_week"]
    feature = {"day_of_the_week": "day_of_week"}.get(feature, feature)
    return getattr(column.dt, feature)


@_get_dt_feature.specialize("polars")
def _get_dt_feature_polars(column, feature):
    if feature == "total_seconds":
        return (column.dt.timestamp(time_unit="ms") / 1000).cast(pl.Float32)
    assert feature in _TIME_LEVELS + ["day_of_the_week"]
    feature = {"day_of_the_week": "weekday"}.get(feature, feature)
    return getattr(column.dt, feature)()


class DatetimeColumnEncoder(BaseEstimator):
    __single_column_transformer__ = True

    def __init__(
        self, resolution="hour", add_day_of_the_week=False, add_total_seconds=True
    ):
        self.resolution = resolution
        self.add_day_of_the_week = add_day_of_the_week
        self.add_total_seconds = add_total_seconds

    def fit_transform(self, column):
        if not sbd.is_anydate(column):
            return NotImplemented
        if self.resolution is None:
            self.extracted_features_ = []
        else:
            idx_level = _TIME_LEVELS.index(self.resolution)
            if _is_date(column):
                idx_level = min(idx_level, _TIME_LEVELS.index("day"))
            self.extracted_features_ = _TIME_LEVELS[: idx_level + 1]
        if self.add_day_of_the_week:
            self.extracted_features_.append("day_of_the_week")
        if self.add_total_seconds:
            self.extracted_features_.append("total_seconds")
        return self.transform(column)

    def transform(self, column):
        name = sbd.name(column)
        all_extracted = []
        for feature in self.extracted_features_:
            extracted = _get_dt_feature(column, feature).rename(f"{name}_{feature}")
            extracted = sbd.to_float32(extracted)
            all_extracted.append(extracted)
        return all_extracted

    def fit(self, column):
        self.fit_transform(column)
        return self


class DatetimeEncoder(TransformerMixin, BaseEstimator):
    """Transforms each datetime column into several numeric columns \
    for temporal features (e.g year, month, day...).

    If the dates are timezone aware, all the features extracted will correspond
    to the provided timezone.

    Parameters
    ----------
    resolution : {"year", "month", "day", "hour", "minute", "second", \
        "microsecond", "nanosecond", None}, default="hour"
        Extract up to this resolution.
        E.g., ``resolution="day"`` generates the features "year", "month",
        "day" only.
        If ``None``, no such feature will be created (but day of the week and \
            total seconds may still be extracted, see below).

    add_day_of_the_week : bool, default=False
        Add day of the week feature as a numerical feature
        from 0 (Monday) to 6 (Sunday).

    add_total_seconds : bool, default=True
        Add the total number of seconds since Epoch.

    Attributes
    ----------
    column_indices_ : list of int
        Indices of the datetime-parsable columns.

    index_to_format_ : dict[int, str]
        Mapping from column indices to their datetime formats.

    index_to_features_ : dict[int, list[str]]
        Dictionary mapping the column names to the list of datetime
        features extracted for each column.

    n_features_out_ : int
        Number of features of the transformed data.

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
    >>> from skrub import DatetimeEncoder
    >>> enc = DatetimeEncoder(add_total_seconds=False)
    >>> X = [['2022-10-15'], ['2021-12-25'], ['2020-05-18'], ['2019-10-15 12:00:00']]
    >>> enc.fit(X)
    DatetimeEncoder(add_total_seconds=False)

    The encoder will output a transformed array
    with four columns ("year", "month", "day", "hour"):

    >>> enc.transform(X)
    array([[2022.,   10.,   15.,    0.],
           [2021.,   12.,   25.,    0.],
           [2020.,    5.,   18.,    0.],
           [2019.,   10.,   15.,   12.]])
    """

    def __init__(
        self,
        *,
        resolution="hour",
        add_day_of_the_week=False,
        add_total_seconds=True,
        parse_string_columns=True,
    ):
        self.resolution = resolution
        self.add_day_of_the_week = add_day_of_the_week
        self.add_total_seconds = add_total_seconds
        self.parse_string_columns = parse_string_columns

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        """Fit the instance to X.

        Select datetime-parsable columns and generate the list of
        datetime feature to extract.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data. Columns that can't be converted into
            ``pandas.DatetimeIndex`` and numerical values will
            be dropped.
        y : None
            Unused, only here for compatibility with scikit-learn.

        Returns
        -------
        DatetimeEncoder
            Fitted DatetimeEncoder instance (self).
        """
        if self.parse_string_columns:
            steps = [Map(ToDatetime())]
        else:
            steps = []
        encoder = DatetimeColumnEncoder(
            resolution=self.resolution,
            add_day_of_the_week=self.add_day_of_the_week,
            add_total_seconds=self.add_total_seconds,
        )
        steps.append(Map(encoder))
        self.pipeline_ = make_pipeline(*steps)
        output = self.pipeline_.fit_transform(X)
        self._output_names = sbd.column_names(output)
        return output

    def transform(self, X):
        return self.pipeline_.transform(X)

    def _more_tags(self):
        """
        Used internally by sklearn to ease the estimator checks.
        """
        return {
            "X_types": ["2darray", "categorical"],
            "allow_nan": True,
            "_xfail_checks": {"check_dtype_object": "Specific datetime error."},
        }
