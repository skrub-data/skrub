import warnings

from pandas._libs.tslibs.parsing import (
    guess_datetime_format as pd_guess_datetime_format,
)
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from . import _selectors as s
from ._dispatch import dispatch
from ._on_each_column import RejectColumn, SingleColumnTransformer
from ._wrap_transformer import wrap_transformer

__all__ = ["ToDatetime", "to_datetime"]

_SAMPLE_SIZE = 30


@dispatch
def _get_time_zone(col):
    raise NotImplementedError()


@_get_time_zone.specialize("pandas", argument_type="Column")
def _get_time_zone_pandas(col):
    tz = col.dt.tz
    if tz is None:
        return None
    if hasattr(tz, "zone"):
        return tz.zone
    return tz.tzname(None)


@_get_time_zone.specialize("polars", argument_type="Column")
def _get_time_zone_polars(col):
    import polars as pl

    if col.dtype == pl.Datetime:
        return col.dtype.time_zone
    return None


@dispatch
def _convert_time_zone(col, time_zone):
    raise NotImplementedError()


@_convert_time_zone.specialize("pandas", argument_type="Column")
def _convert_time_zone_pandas(col, time_zone):
    is_localized = _get_time_zone(col) is not None
    if is_localized:
        if time_zone is None:
            return col.dt.tz_convert(None)
        else:
            return col.dt.tz_convert(time_zone)
    else:
        if time_zone is None:
            return col
        else:
            return col.dt.tz_localize("UTC").dt.tz_convert(time_zone)


@_convert_time_zone.specialize("polars", argument_type="Column")
def _convert_time_zone_polars(col, time_zone):
    import polars as pl

    is_localized = _get_time_zone(col) is not None
    if is_localized:
        if time_zone is None:
            return col.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
        else:
            return col.dt.convert_time_zone(time_zone)
    else:
        if time_zone is None:
            return col
        else:
            if col.dtype == pl.Date:
                col = col.cast(pl.Datetime)
            return col.dt.replace_time_zone("UTC").dt.convert_time_zone(time_zone)


class ToDatetime(SingleColumnTransformer):
    """
    Parse datetimes represented as strings and return ``Datetime`` columns.

    An input column is converted to a column with dtype Datetime if possible,
    and rejected by raising a ``RejectColumn`` exception otherwise. Only Date,
    Datetime, String, and pandas object columns are handled, other dtypes are
    rejected with ``RejectColumn``.

    Once a column is accepted, outputs of ``transform`` always have the same
    Datetime dtype (including resolution and time zone). Once the transformer
    is fitted, entries that fail to be converted during subsequent calls to
    ``transform`` are replaced with nulls.

    Parameters
    ----------
    format : str or None, optional, default=None
        Format to use for parsing dates that are stored as strings, e.g.
        ``"%Y-%m-%dT%H:%M%S"``.
        If not specfied, the format is inferred from the data when possible.
        When doing so, for dates presented as 01/02/2003, it is usually
        possible to infer from the data whether the month comes first (USA
        convention) or the day comes first, ie ``"%m/%d/%Y"`` vs
        ``"%d/%m/%Y"``. In the odd chance that all the sampled dates land
        before the 13th day of the month and that both conventions are
        plausible, the USA convention (month first) is chosen.

    Attributes
    ----------
    format_ : str or None
        Detected format. If the transformer was fitted on a column that already had
        a Datetime dtype, the ``format_`` is None. Otherwise it is the
        format that was detected when parsing the string column. If the parameter
        ``format`` was provided, it is the only one that the transformer
        attempts to use so in that caset ``format_`` is either ``None`` or
        equal to ``format``.

    output_dtype_ : data type
        The output dtype, which includes information about the time resolution and
        time zone.

    output_time_zone_ : str or None
        The time zone of the transformed column. If the output is time zone naive it
        is ``None``; otherwise it is the name of the time zone such as ``UTC`` or
        ``Europe/Paris``.

    Examples
    --------
    >>> import pandas as pd

    >>> s = pd.Series(["2024-05-05T13:17:52", None, "2024-05-07T13:17:52"], name="when")
    >>> s
    0    2024-05-05T13:17:52
    1                   None
    2    2024-05-07T13:17:52
    Name: when, dtype: object

    >>> from skrub._to_datetime import ToDatetime

    >>> to_dt = ToDatetime()
    >>> to_dt.fit_transform(s)
    0   2024-05-05 13:17:52
    1                   NaT
    2   2024-05-07 13:17:52
    Name: when, dtype: datetime64[ns]

    The attributes ``format_``, ``output_dtype_``, ``output_time_zone_``
    record information about the conversion result.

    >>> to_dt.format_
    '%Y-%m-%dT%H:%M:%S'
    >>> to_dt.output_dtype_
    dtype('<M8[ns]')
    >>> to_dt.output_time_zone_ is None
    True

    If we provide the datetime format, it is used and columns that do not conform to
    it are rejected.

    >>> ToDatetime(format="%Y-%m-%dT%H:%M:%S").fit_transform(s)
    0   2024-05-05 13:17:52
    1                   NaT
    2   2024-05-07 13:17:52
    Name: when, dtype: datetime64[ns]

    >>> ToDatetime(format="%d/%m/%Y").fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Failed to convert column 'when' to datetimes using the format '%d/%m/%Y'.

    Columns that already have ``Datetime`` ``dtype`` are not modified (but
    they are accepted); for those columns the provided format, if any, is ignored.

    >>> s = pd.to_datetime(s).dt.tz_localize("Europe/Paris")
    >>> s
    0   2024-05-05 13:17:52+02:00
    1                         NaT
    2   2024-05-07 13:17:52+02:00
    Name: when, dtype: datetime64[ns, Europe/Paris]
    >>> to_dt.fit_transform(s) is s
    True

    In that case the ``format_`` is ``None``.

    >>> to_dt.format_ is None
    True
    >>> to_dt.output_dtype_
    datetime64[ns, Europe/Paris]
    >>> to_dt.output_time_zone_
    'Europe/Paris'

    Columns that have a different ``dtype`` than strings, pandas objects, or
    datetimes are rejected.

    >>> s = pd.Series([2020, 2021, 2022], name="year")
    >>> to_dt.fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Column 'year' does not contain strings.

    String columns that do not appear to contain datetimes or for some other reason
    fail to be converted are also rejected.

    >>> s = pd.Series(["2024-05-07T13:36:27", "yesterday"], name="when")
    >>> to_dt.fit_transform(s)
    Traceback (most recent call last):
        ...
    skrub._on_each_column.RejectColumn: Could not find a datetime format for column 'when'.

    Once ``ToDatetime`` was successfully fitted, ``transform`` will always try to
    parse datetimes with the same format and output the same ``dtype``. Entries that
    fail to be converted result in a null value:

    >>> s = pd.Series(["2024-05-05T13:17:52", None, "2024-05-07T13:17:52"], name="when")
    >>> to_dt = ToDatetime().fit(s)
    >>> to_dt.transform(s)
    0   2024-05-05 13:17:52
    1                   NaT
    2   2024-05-07 13:17:52
    Name: when, dtype: datetime64[ns]
    >>> s = pd.Series(["05/05/2024", None, "07/05/2024"], name="when")
    >>> to_dt.transform(s)
    0   NaT
    1   NaT
    2   NaT
    Name: when, dtype: datetime64[ns]

    **Time zones**

    During ``fit``, parsing strings that contain fixed offsets results in datetimes
    in UTC. Mixed offsets are supported and will all be converted to UTC.

    >>> s = pd.Series(["2020-01-01T04:00:00+02:00", "2020-01-01T04:00:00+03:00"])
    >>> to_dt.fit_transform(s)
    0   2020-01-01 02:00:00+00:00
    1   2020-01-01 01:00:00+00:00
    dtype: datetime64[ns, UTC]
    >>> to_dt.format_
    '%Y-%m-%dT%H:%M:%S%z'
    >>> to_dt.output_time_zone_
    'UTC'

    Strings with no timezone indication result in naive datetimes:

    >>> s = pd.Series(["2020-01-01T04:00:00", "2020-01-01T04:00:00"])
    >>> to_dt.fit_transform(s)
    0   2020-01-01 04:00:00
    1   2020-01-01 04:00:00
    dtype: datetime64[ns]
    >>> to_dt.output_time_zone_ is None
    True

    During ``transform``, outputs are cast to the same ``dtype`` that was found
    during ``fit``. This includes the timezone, which is converted if necessary.

    >>> s_paris = pd.to_datetime(
    ...     pd.Series(["2024-05-07T14:24:49", "2024-05-06T14:24:49"])
    ... ).dt.tz_localize("Europe/Paris")
    >>> s_paris
    0   2024-05-07 14:24:49+02:00
    1   2024-05-06 14:24:49+02:00
    dtype: datetime64[ns, Europe/Paris]
    >>> to_dt = ToDatetime().fit(s_paris)
    >>> to_dt.output_dtype_
    datetime64[ns, Europe/Paris]

    Here our converter is set to output datetimes with nanosecond resolution,
    localized in "Europe/Paris".

    We may have a column in a different timezone:

    >>> s_london = s_paris.dt.tz_convert("Europe/London")
    >>> s_london
    0   2024-05-07 13:24:49+01:00
    1   2024-05-06 13:24:49+01:00
    dtype: datetime64[ns, Europe/London]

    Here the timezone is "Europe/London" and the times are offset by 1 hour. During
    ``transform`` datetimes will be converted to the original dtype and the
    "Europe/Paris" timezone:

    >>> to_dt.transform(s_london)
    0   2024-05-07 14:24:49+02:00
    1   2024-05-06 14:24:49+02:00
    dtype: datetime64[ns, Europe/Paris]

    Moreover, we may have to transform a timezone-naive column whereas the
    transformer was fitted on a timezone-aware column. Note that is somewhat a
    corner case unlikely to happen in practice if the inputs to ``fit`` and
    ``transform`` come from the same dataframe.

    >>> s_naive = s_paris.dt.tz_convert(None)
    >>> s_naive
    0   2024-05-07 12:24:49
    1   2024-05-06 12:24:49
    dtype: datetime64[ns]

    In this case, we make the arbitrary choice to assume that the timezone-naive
    datetimes are in UTC.

    >>> to_dt.transform(s_naive)
    0   2024-05-07 14:24:49+02:00
    1   2024-05-06 14:24:49+02:00
    dtype: datetime64[ns, Europe/Paris]

    Conversely, a transformer fitted on a timezone-naive column can convert
    timezone-aware columns. Here also, we assume the naive datetimes were in UTC.

    >>> to_dt = ToDatetime().fit(s_naive)
    >>> to_dt.transform(s_london)
    0   2024-05-07 12:24:49
    1   2024-05-06 12:24:49
    dtype: datetime64[ns]

    **``%d/%m/%Y`` vs ``%m/%d/%Y``**

    When parsing strings in one of the formats above, ``ToDatetime`` tries to guess
    if the month comes first (USA convention) or the day (rest of the world) from
    the data.

    >>> s = pd.Series(["05/23/2024"])
    >>> to_dt.fit_transform(s)
    0   2024-05-23
    dtype: datetime64[ns]
    >>> to_dt.format_
    '%m/%d/%Y'

    Here we could infer ``'%m/%d/%Y'`` because there are not 23 months in a year.
    Similarly,

    >>> s = pd.Series(["23/05/2024"])
    >>> to_dt.fit_transform(s)
    0   2024-05-23
    dtype: datetime64[ns]
    >>> to_dt.format_
    '%d/%m/%Y'

    In the case it cannot be inferred, the USA convention is used:

    >>> s = pd.Series(["03/05/2024"])
    >>> to_dt.fit_transform(s)
    0   2024-03-05
    dtype: datetime64[ns]
    >>> to_dt.format_
    '%m/%d/%Y'

    If the days are randomly distributed and the fitting data large enough, it is
    somewhat unlikely that all days would be below 12 so the inferred format should
    often be correct. To be sure, one can specify the ``format`` in the
    constructor.
    """  # noqa: E501

    def __init__(self, format=None):
        self.format = format

    def fit_transform(self, column, y=None):
        """Fit the encoder and transform a column.

        Parameters
        ----------
        column : pandas or polars Series
            The input to transform.

        y : None
            Ignored.

        Returns
        -------
        transformed : pandas or polars Series.
            The input transformed to Datetime.
        """
        del y
        if sbd.is_any_date(column):
            self.format_ = None
            self.output_dtype_ = sbd.dtype(column)
            self.output_time_zone_ = _get_time_zone(column)
            return column
        if not (sbd.is_pandas_object(column) or sbd.is_string(column)):
            raise RejectColumn(f"Column {sbd.name(column)!r} does not contain strings.")

        datetime_format = self._get_datetime_format(column)
        if datetime_format is None:
            raise RejectColumn(
                f"Could not find a datetime format for column {sbd.name(column)!r}."
            )

        self.format_ = datetime_format
        try:
            as_datetime = sbd.to_datetime(column, format=self.format_, strict=True)
        except Exception as e:
            raise RejectColumn(
                f"Failed to convert column {sbd.name(column)!r} to datetimes "
                f"using the format {self.format_!r}."
            ) from e
        self.output_dtype_ = sbd.dtype(as_datetime)
        self.output_time_zone_ = _get_time_zone(as_datetime)
        return as_datetime

    def transform(self, column):
        """Transform a column.

        Parameters
        ----------
        column : pandas or polars Series
            The input to transform.

        Returns
        -------
        transformed : pandas or polars Series.
            The input transformed to Datetime.
        """
        check_is_fitted(self, "format_")
        column = sbd.to_datetime(column, format=self.format_, strict=False)
        column = _convert_time_zone(column, self.output_time_zone_)
        return sbd.cast(column, self.output_dtype_)

    def _get_datetime_format(self, column):
        if self.format is not None:
            return self.format
        not_null = sbd.drop_nulls(column)
        sample = sbd.sample(
            not_null, n=min(_SAMPLE_SIZE, sbd.shape(not_null)[0]), seed=0
        )
        if not sbd.is_string(sample):
            return None
        return _guess_datetime_format(sample)


def _guess_datetime_format(column):
    """Guess the format of a column of datetimes represented as strings.

    When no format is found that can be successfully applied to the whole
    column, return ``None``. When both day-first and month-first formats are
    possible, month-first is kept.
    """
    column = sbd.to_pandas(column)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        month_first_formats = column.apply(
            pd_guess_datetime_format, dayfirst=False
        ).unique()
        if len(month_first_formats) == 1 and month_first_formats[0] is not None:
            return str(month_first_formats[0])

        day_first_formats = column.apply(
            pd_guess_datetime_format, dayfirst=True
        ).unique()
        if len(day_first_formats) == 1 and day_first_formats[0] is not None:
            return str(day_first_formats[0])

    return None


@dispatch
def to_datetime(data, format=None):
    """Convert DataFrame or column to Datetime dtype.

    Parameters
    ----------
    data : dataframe or column
        The dataframe or column to transform.

    format : str or None, optional, default=None
        Format string to use to parse datetime strings.
        See the reference documentation for format codes:
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes .

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub import to_datetime
    >>> X = pd.DataFrame(dict(a=[1, 2], b=["01/02/2021", "21/02/2021"]))
    >>> X
       a           b
    0  1  01/02/2021
    1  2  21/02/2021
    >>> to_datetime(X)
       a          b
    0  1 2021-02-01
    1  2 2021-02-21
    """  # noqa: E501
    raise TypeError(
        "Input to skrub.to_datetime must be a pandas or polars Series or DataFrame."
        f" Got {type(data)}."
    )


@to_datetime.specialize("pandas", argument_type="DataFrame")
@to_datetime.specialize("polars", argument_type="DataFrame")
def _to_datetime_dataframe(df, format=None):
    return wrap_transformer(
        ToDatetime(format=format), s.all(), allow_reject=True
    ).fit_transform(df)


@to_datetime.specialize("pandas", argument_type="Column")
@to_datetime.specialize("polars", argument_type="Column")
def _to_datetime_column(column, format=None):
    try:
        result = ToDatetime(format=format).fit_transform(column)
    except RejectColumn:
        return column
    return result
