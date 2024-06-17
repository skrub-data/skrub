from datetime import datetime, timezone

import pandas as pd
from sklearn.utils.validation import check_is_fitted

try:
    import polars as pl
except ImportError:
    pass

from . import _dataframe as sbd
from ._dispatch import dispatch
from ._on_each_column import RejectColumn, SingleColumnTransformer

__all__ = ["DatetimeEncoder"]

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


@dispatch
def _is_date(col):
    raise NotImplementedError()


@_is_date.specialize("pandas", argument_type="Column")
def _is_date_pandas(col):
    col = sbd.drop_nulls(col)
    return (col.dt.date == col).all()


@_is_date.specialize("polars", argument_type="Column")
def _is_date_polars(col):
    return (col.dt.date() == col).all()


@dispatch
def _get_dt_feature(col, feature):
    raise NotImplementedError()


@_get_dt_feature.specialize("pandas", argument_type="Column")
def _get_dt_feature_pandas(col, feature):
    if feature == "total_seconds":
        if col.dt.tz is None:
            epoch = datetime(1970, 1, 1)
        else:
            epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
        return ((col - epoch) / pd.Timedelta("1s")).astype("float32")
    if feature == "weekday":
        return col.dt.day_of_week + 1
    assert feature in _TIME_LEVELS
    return getattr(col.dt, feature)


@_get_dt_feature.specialize("polars", argument_type="Column")
def _get_dt_feature_polars(col, feature):
    if feature == "total_seconds":
        return (col.dt.timestamp(time_unit="ms") / 1000).cast(pl.Float32)
    assert feature in _TIME_LEVELS + ["weekday"]
    return getattr(col.dt, feature)()


class DatetimeEncoder(SingleColumnTransformer):
    """
    Extract temporal features such as month, day of the week, … from a datetime column.

    All extracted features are provided as float32 columns.

    No timezone conversion is performed: if the input column is timezone aware, the
    extracted features will be in the column's timezone.

    An input column that does not have a Date or Datetime dtype will be
    rejected by raising a ``RejectColumn`` exception. See ``ToDatetime`` for
    converting strings to proper datetimes. **Note:** the ``TableVectorizer``
    only sends datetime columns to its ``datetime_encoder``. Therefore it is
    always safe to use a ``DatetimeEncoder`` as the ``TableVectorizer``'s
    ``datetime_encoder`` parameter.

    Parameters
    ----------
    resolution : str or None, default="hour"
        If a string, extract up to this resolution. Must be "year", "month",
        "day", "hour", "minute", "second", "microsecond", or "nanosecond". For
        example, ``resolution="day"`` generates the features "year", "month",
        and "day" only. If the input column contains dates with no time
        information, time features ("hour", "minute", … ) are never extracted.
        If ``None``, the features listed above are not extracted (but day of
        the week and total seconds may still be extracted, see below).

    add_weekday : bool, default=False
        Extract the day of the week as a numerical feature from 1 (Monday) to 7
        (Sunday).

    add_total_seconds : bool, default=True
        Add the total number of seconds since the Unix epoch (00:00:00 UTC on 1
        January 1970).

    Attributes
    ----------
    extracted_features_ : list of strings
        The features that are extracted, a subset of ["year", …, "nanosecond",
        "weekday", "total_seconds"]

    See Also
    --------
    ToDatetime :
        Convert strings to datetimes.

    Examples
    --------
    >>> import pandas as pd

    >>> login = pd.to_datetime(
    ...     pd.Series(
    ...         ["2024-05-13T12:05:36", None, "2024-05-15T13:46:02"], name="login")
    ... )
    >>> login
    0   2024-05-13 12:05:36
    1                   NaT
    2   2024-05-15 13:46:02
    Name: login, dtype: datetime64[ns]
    >>> from skrub import DatetimeEncoder

    >>> DatetimeEncoder().fit_transform(login)
       login_year  login_month  login_day  login_hour  login_total_seconds
    0      2024.0          5.0       13.0        12.0         1.715602e+09
    1         NaN          NaN        NaN         NaN                  NaN
    2      2024.0          5.0       15.0        13.0         1.715781e+09

    We can ask for a finer resolution:

    >>> DatetimeEncoder(resolution='second', add_total_seconds=False).fit_transform(
    ...     login
    ... )
       login_year  login_month  login_day  login_hour  login_minute  login_second
    0      2024.0          5.0       13.0        12.0           5.0          36.0
    1         NaN          NaN        NaN         NaN           NaN           NaN
    2      2024.0          5.0       15.0        13.0          46.0           2.0

    We can also ask for the day of the week. The week starts at 1 on Monday and ends
    at 7 on Sunday. This is consistent with the ISO week date system
    (https://en.wikipedia.org/wiki/ISO_week_date), the standard library
    ``datetime.isoweekday()`` and polars ``weekday``, but not with pandas
    ``day_of_week``, which counts days from 0.

    >>> login.dt.strftime('%A = %w')
    0       Monday = 1
    1              NaN
    2    Wednesday = 3
    Name: login, dtype: object
    >>> login.dt.day_of_week
    0    0.0
    1    NaN
    2    2.0
    Name: login, dtype: float64
    >>> DatetimeEncoder(add_weekday=True, add_total_seconds=False).fit_transform(login)
       login_year  login_month  login_day  login_hour  login_weekday
    0      2024.0          5.0       13.0        12.0            1.0
    1         NaN          NaN        NaN         NaN            NaN
    2      2024.0          5.0       15.0        13.0            3.0

    When a column contains only dates without time information, the time features
    are discarded, regardless of ``resolution``.

    >>> birthday = pd.to_datetime(
    ...     pd.Series(['2024-04-14', '2024-05-15'], name='birthday')
    ... )
    >>> encoder = DatetimeEncoder(resolution='second')
    >>> encoder.fit_transform(birthday)
       birthday_year  birthday_month  birthday_day  birthday_total_seconds
    0         2024.0             4.0          14.0            1.713053e+09
    1         2024.0             5.0          15.0            1.715731e+09
    >>> encoder.extracted_features_
    ['year', 'month', 'day', 'total_seconds']

    (The number of seconds since Epoch can still be extracted but not "hour",
    "minute", etc.)

    Non-datetime columns are rejected by raising a ``RejectColumn`` exception.

    >>> s = pd.Series(['2024-04-14', '2024-05-15'], name='birthday')
    >>> s
    0    2024-04-14
    1    2024-05-15
    Name: birthday, dtype: object
    >>> DatetimeEncoder().fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Column 'birthday' does not have Date or Datetime dtype.

    :class:`ToDatetime`: can be used for converting strings to datetimes.

    >>> from skrub import ToDatetime
    >>> from sklearn.pipeline import make_pipeline
    >>> make_pipeline(ToDatetime(), DatetimeEncoder()).fit_transform(s)
       birthday_year  birthday_month  birthday_day  birthday_total_seconds
    0         2024.0             4.0          14.0            1.713053e+09
    1         2024.0             5.0          15.0            1.715731e+09

    **Time zones**

    If the input column has a time zone, the extracted features are in this timezone.

    >>> login = pd.to_datetime(
    ...     pd.Series(
    ...         ["2024-05-13T12:05:36", None, "2024-05-15T13:46:02"], name="login")
    ... ).dt.tz_localize('Europe/Paris')
    >>> encoder = DatetimeEncoder()
    >>> encoder.fit_transform(login)['login_hour']
    0    12.0
    1     NaN
    2    13.0
    Name: login_hour, dtype: float32

    No special care is taken to convert inputs to ``transform`` to the same time
    zone as the column the encoder was fitted on. The features are always in the
    time zone of the input.

    >>> login_sp = login.dt.tz_convert('America/Sao_Paulo')
    >>> login_sp
    0   2024-05-13 07:05:36-03:00
    1                         NaT
    2   2024-05-15 08:46:02-03:00
    Name: login, dtype: datetime64[ns, America/Sao_Paulo]
    >>> encoder.transform(login_sp)['login_hour']
    0    7.0
    1    NaN
    2    8.0
    Name: login_hour, dtype: float32

    To ensure datetime columns are in a consistent timezones, use ``ToDatetime``.

    >>> encoder = make_pipeline(ToDatetime(), DatetimeEncoder())
    >>> encoder.fit_transform(login)['login_hour']
    0    12.0
    1     NaN
    2    13.0
    Name: login_hour, dtype: float32
    >>> encoder.transform(login_sp)['login_hour']
    0    12.0
    1     NaN
    2    13.0
    Name: login_hour, dtype: float32

    Here we can see the input to ``transform`` has been converted back to the
    timezone used during ``fit`` and that we get the same result for "hour".
    """  # noqa: E501

    def __init__(self, resolution="hour", add_weekday=False, add_total_seconds=True):
        self.resolution = resolution
        self.add_weekday = add_weekday
        self.add_total_seconds = add_total_seconds

    def fit_transform(self, column, y=None):
        """Fit the encoder and transform a column.

        Parameters
        ----------
        column : pandas or polars Series with dtype Date or Datetime
            The input to transform.

        y : None
            Ignored.

        Returns
        -------
        transformed : DataFrame
            The extracted features.
        """
        del y
        self._check_params()
        if not sbd.is_any_date(column):
            raise RejectColumn(
                f"Column {sbd.name(column)!r} does not have Date or Datetime dtype."
            )
        if self.resolution is None:
            self.extracted_features_ = []
        else:
            idx_level = _TIME_LEVELS.index(self.resolution)
            if _is_date(column):
                idx_level = min(idx_level, _TIME_LEVELS.index("day"))
            self.extracted_features_ = _TIME_LEVELS[: idx_level + 1]
        if self.add_total_seconds:
            self.extracted_features_.append("total_seconds")
        if self.add_weekday:
            self.extracted_features_.append("weekday")
        return self.transform(column)

    def transform(self, column):
        """Transform a column.

        Parameters
        ----------
        column : pandas or polars Series with dtype Date or Datetime
            The input to transform.

        Returns
        -------
        transformed : DataFrame
            The extracted features.
        """
        check_is_fitted(self, "extracted_features_")
        name = sbd.name(column)
        all_extracted = []
        for feature in self.extracted_features_:
            extracted = _get_dt_feature(column, feature).rename(f"{name}_{feature}")
            extracted = sbd.to_float32(extracted)
            all_extracted.append(extracted)
        return sbd.make_dataframe_like(column, all_extracted)

    def _check_params(self):
        allowed = _TIME_LEVELS + [None]
        if self.resolution not in allowed:
            raise ValueError(
                f"'resolution' options are {allowed}, got {self.resolution!r}."
            )
