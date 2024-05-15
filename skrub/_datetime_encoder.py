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
from ._dispatch import dispatch
from ._on_each_column import RejectColumn, SingleColumnTransformer
from ._to_datetime import ToDatetime
from ._wrap_transformer import wrap_transformer

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
def _is_date(column):
    raise NotImplementedError()


@_is_date.specialize("pandas")
def _is_date_pandas(column):
    column = sbd.drop_nulls(column)
    return (column.dt.date == column).all()


@_is_date.specialize("polars")
def _is_date_polars(column):
    return (column.dt.date() == column).all()


@dispatch
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
    if feature == "day_of_the_week":
        return column.dt.day_of_week + 1
    assert feature in _TIME_LEVELS
    return getattr(column.dt, feature)


@_get_dt_feature.specialize("polars")
def _get_dt_feature_polars(column, feature):
    if feature == "total_seconds":
        return (column.dt.timestamp(time_unit="ms") / 1000).cast(pl.Float32)
    assert feature in _TIME_LEVELS + ["day_of_the_week"]
    feature = {"day_of_the_week": "weekday"}.get(feature, feature)
    return getattr(column.dt, feature)()


class EncodeDatetime(SingleColumnTransformer):
    """
    Extract temporal features such as month, day of the week, … from a datetime column.

    All extracted features are provided as float32 columns.

    No timezone conversion is performed: if the input column is timezone aware, the
    extracted features will be in the column's timezone.

    An input column that does not have a Date or Datetime dtype will be rejected by
    raising a ``RejectColumn`` exception. See ``ToDatetime`` for converting strings
    to proper datetimes.

    Parameters
    ----------
    resolution : {"year", "month", "day", "hour", "minute", "second", "microsecond", "nanosecond", None}, default="hour"
        Extract up to this resolution. E.g., ``resolution="day"`` generates the
        features "year", "month", "day" only. If the input column contains dates
        with no time information, time features ("hour", "minute", … ) are never
        extracted. If ``None``, the features listed above are not extracted (but day
        of the week and total seconds may still be extracted, see below).

    add_day_of_the_week : bool, default=False
        Extract the day of the week as a numerical feature from 1 (Monday) to 7 (Sunday).

    add_total_seconds : bool, default=True
        Add the total number of seconds since Epoch.

    Attributes
    ----------
    extracted_features_ : list of strings
        The features that are extracted, a subset of ["year", …, "nanosecond",
        "day_of_the_week", "total_seconds"]

    See Also
    --------
    ToDatetime :
        Convert strings to datetimes.

    Examples
    --------
    >>> import pandas as pd

    >>> login = pd.to_datetime(
    ...     pd.Series(["2024-05-13T12:05:36", None, "2024-05-15T13:46:02"], name="login")
    ... )
    >>> login
    0   2024-05-13 12:05:36
    1                   NaT
    2   2024-05-15 13:46:02
    Name: login, dtype: datetime64[ns]
    >>> from skrub import EncodeDatetime

    >>> EncodeDatetime().fit_transform(login)
       login_year  login_month  login_day  login_hour  login_total_seconds
    0      2024.0          5.0       13.0        12.0         1.715602e+09
    1         NaN          NaN        NaN         NaN                  NaN
    2      2024.0          5.0       15.0        13.0         1.715781e+09

    We can ask for a finer resolution:

    >>> EncodeDatetime(resolution='second', add_total_seconds=False).fit_transform(login)
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
    >>> EncodeDatetime(add_day_of_the_week=True, add_total_seconds=False).fit_transform(login)
       login_year  login_month  login_day  login_hour  login_day_of_the_week
    0      2024.0          5.0       13.0        12.0                    1.0
    1         NaN          NaN        NaN         NaN                    NaN
    2      2024.0          5.0       15.0        13.0                    3.0

    When a column contains only dates without time information, the time features
    are discarded, regardless of ``resolution``.

    >>> birthday = pd.to_datetime(pd.Series(['2024-04-14', '2024-05-15'], name='birthday'))
    >>> encoder = EncodeDatetime(resolution='second')
    >>> encoder.fit_transform(birthday)
       birthday_year  birthday_month  birthday_day  birthday_total_seconds
    0         2024.0             4.0          14.0            1.713053e+09
    1         2024.0             5.0          15.0            1.715731e+09
    >>> encoder.extracted_features_
    ['year', 'month', 'day', 'total_seconds']

    (The number of seconds since Epoch can still be extracted but not "hour", "minute", etc.)

    Non-datetime columns are rejected by raising a ``RejectColumn`` exception.

    >>> s = pd.Series(['2024-04-14', '2024-05-15'], name='birthday')
    >>> s
    0    2024-04-14
    1    2024-05-15
    Name: birthday, dtype: object
    >>> EncodeDatetime().fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Column 'birthday' does not have Date or Datetime dtype.

    ToDatetime can be used for converting strings to datetimes.

    >>> from skrub import ToDatetime
    >>> from sklearn.pipeline import make_pipeline
    >>> make_pipeline(ToDatetime(), EncodeDatetime()).fit_transform(s)
       birthday_year  birthday_month  birthday_day  birthday_total_seconds
    0         2024.0             4.0          14.0            1.713053e+09
    1         2024.0             5.0          15.0            1.715731e+09

    **Time zones**

    If the input column has a time zone, the extracted features are in this timezone.

    >>> login = pd.to_datetime(
    ...     pd.Series(["2024-05-13T12:05:36", None, "2024-05-15T13:46:02"], name="login")
    ... ).dt.tz_localize('Europe/Paris')
    >>> encoder = EncodeDatetime()
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

    >>> encoder = make_pipeline(ToDatetime(), EncodeDatetime())
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
    """

    def __init__(
        self, resolution="hour", add_day_of_the_week=False, add_total_seconds=True
    ):
        self.resolution = resolution
        self.add_day_of_the_week = add_day_of_the_week
        self.add_total_seconds = add_total_seconds

    def fit_transform(self, column, y=None):
        """Fit the encoder and transform a colum.

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
        if self.add_day_of_the_week:
            self.extracted_features_.append("day_of_the_week")
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


class DatetimeEncoder(TransformerMixin, BaseEstimator, auto_wrap_output_keys=()):
    """Transforms each datetime column into several numeric columns \
    for temporal features (e.g. year, month, day...).

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
        from 1 (Monday) to 7 (Sunday).

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
            self._to_datetime = wrap_transformer(
                ToDatetime(), s.all(), allow_reject=True
            )
            steps.append(self._to_datetime)
        column_encoder = EncodeDatetime(
            resolution=self.resolution,
            add_day_of_the_week=self.add_day_of_the_week,
            add_total_seconds=self.add_total_seconds,
        )
        self._encoder = wrap_transformer(column_encoder, s.any_date())
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
