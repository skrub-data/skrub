from datetime import datetime, timezone

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
from . import _selectors as s
from ._check_input import CheckInputDataFrame
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
    column = sbd.drop_nulls(column)
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
            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
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

    def _check_params(self):
        allowed = _TIME_LEVELS + [None]
        if self.resolution not in allowed:
            raise ValueError(
                f"'resolution' options are {allowed}, got {self.resolution!r}."
            )

    def fit_transform(self, column):
        self._check_params()
        if not sbd.is_anydate(column):
            return NotImplemented
        if self.resolution is None:
            self.extracted_features_ = []
        else:
            idx_level = _TIME_LEVELS.index(self.resolution)
            if _is_date(column):
                idx_level = min(idx_level, _TIME_LEVELS.index("day"))
            self.extracted_features_ = _TIME_LEVELS[: idx_level + 1]
        if self.add_total_seconds:
            self.extracted_features_.append("total_seconds")
        if self.add_day_of_the_week:
            self.extracted_features_.append("day_of_the_week")
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


class DatetimeEncoder(TransformerMixin, BaseEstimator, auto_wrap_output_keys=()):
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

    parse_string_columns : bool, default=True
        Attempt to convert columns converting strings (such as "02/02/2024") to
        datetimes, before extracting features. If False, only columns with
        dtype Date or Datetime are considered.

    Attributes
    ----------
    input_to_outputs_: dict
        Maps each column name in the input dataframe to the list of
        corresponding column names (features extracted from the input column)
        in the output dataframe.
        Only contains the names of columns that were actually processed by this
        transformers, that is, datetime columns.

    all_outputs_: list[str]
        The column names in the transformer's output (all column names,
        including those that were not modified).

    n_features_out_ : int
        Number of features of the transformer' output.

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
    >>> import pandas as pd
    >>> from skrub import DatetimeEncoder
    >>> enc = DatetimeEncoder(add_total_seconds=False)
    >>> X = pd.DataFrame(
    ...     {"birthday": ["2022-10-15", "2021-12-25", "2020-05-18", "2019-10-15"]}
    ... )
    >>> enc.fit(X)
    DatetimeEncoder(add_total_seconds=False)

    The encoder will output a transformed array
    with four columns ("year", "month", "day", "hour"):

    >>> enc.transform(X)
       birthday_year  birthday_month  birthday_day
    0         2022.0            10.0          15.0
    1         2021.0            12.0          25.0
    2         2020.0             5.0          18.0
    3         2019.0            10.0          15.0
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
        """Fit the encoder to a dataframe.

        Parameters
        ----------
        X : DataFrame
            Input data. Columns that are not of the dtype Date or Datetime (and
            cannot be converted to Datetime, when ``parse_string_columns`` is
            ``True``), will be passed through unchanged.
        y : None
            Unused, only here for compatibility with scikit-learn.

        Returns
        -------
        DatetimeEncoder
            The fitted encoder.
        """
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        """Fit the encoder to a dataframe and transform the dataframe.

        Parameters
        ----------
        X : DataFrame
            Input data. Columns that are not of the dtype Date or Datetime (and
            cannot be converted to such, when ``parse_string_columns`` is
            ``True``), will be passed through unchanged.
        y : None
            Unused, only here for compatibility with scikit-learn.

        Returns
        -------
        DataFrame
            The transformed dataframe.
        """
        steps = [CheckInputDataFrame()]
        if self.parse_string_columns:
            self._to_datetime = s.all().use(ToDatetime())
            steps.append(self._to_datetime)
        column_encoder = DatetimeColumnEncoder(
            resolution=self.resolution,
            add_day_of_the_week=self.add_day_of_the_week,
            add_total_seconds=self.add_total_seconds,
        )
        self._encoder = s.all().use(column_encoder)
        steps.append(self._encoder)
        self.pipeline_ = make_pipeline(*steps)
        output = self.pipeline_.fit_transform(X)
        if self.parse_string_columns:
            self.datetime_formats_ = {
                c: t.datetime_format_
                for (c, t) in self._to_datetime.transformers_.items()
            }
        self.all_outputs_ = sbd.column_names(output)
        self.n_features_out_ = len(self.all_outputs_)
        self.input_to_outputs_ = self._encoder.input_to_outputs_
        return output

    def transform(self, X):
        """Transform a dataframe.

        Parameters
        ----------
        X : DataFrame
            Input data. Columns that are not of the dtype Date or Datetime (and
            cannot be converted to Datetime, when ``parse_string_columns`` is
            ``True``), will be passed through unchanged.

        Returns
        -------
        DataFrame
            The transformed dataframe.
        """
        return self.pipeline_.transform(X)

    def get_feature_names_out(self):
        """Return the column names of the output of ``transform`` as a list of strings.

        Returns
        -------
        list of strings
            The column names.
        """
        return self.all_outputs_

    def _more_tags(self):
        """
        Used internally by sklearn to ease the estimator checks.
        """
        return {
            "X_types": ["2darray", "categorical"],
            "allow_nan": True,
            "_xfail_checks": {"check_dtype_object": "Specific datetime error."},
        }
