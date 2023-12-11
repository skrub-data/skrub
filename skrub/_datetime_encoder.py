import warnings
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from pandas._libs.tslibs.parsing import guess_datetime_format
from pandas.api.types import is_datetime64_any_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.fixes import parse_version
from sklearn.utils.validation import check_is_fitted, check_random_state

from ._dataframe._namespace import get_df_namespace

WORD_TO_ALIAS = {
    "year": "Y",
    "month": "M",
    "day": "D",
    "hour": "H",
    "minute": "min",
    "second": "S",
    "microsecond": "us",
    "nanosecond": "N",
}
TIME_LEVELS = list(WORD_TO_ALIAS)


def _is_pandas_format_mixed_available():
    pandas_version = pd.__version__
    min_pandas_version = "2.0.0"
    return parse_version(min_pandas_version) < parse_version(pandas_version)


MIXED_FORMAT = "mixed" if _is_pandas_format_mixed_available() else None


def to_datetime(
    X,
    errors="coerce",
    random_state=None,
    **kwargs,
):
    """Convert the columns of a dataframe or 2d array into a datetime representation.

    This function augments :func:`pandas.to_datetime` by supporting dataframes
    and 2d array inputs. It only attempts to convert columns whose dtype are
    object or string. Numeric columns are skip and preserved in the output.

    Use the 'format' keyword to force a specific datetime format. See more details in
    the parameters section.

    Parameters
    ----------
    X : Pandas or Polars dataframe, 2d-array or any input accepted \
    by ``pd.to_datetime``
        The object to convert to a datetime.

    errors : {'coerce', 'raise'}, default 'coerce'
        When set to 'raise', errors will be raised only when the following conditions
        are satisfied, for each column ``X_col``:
        - After converting to numpy, the column dtype is np.object_ or np.str_
        - Each entry of the column is datetime-parsable, i.e.
          ``pd.to_datetime(X_col, format="mixed")`` doesn't raise an error.
          This step is conservative, because e.g.
          ``["2020-01-01", "hello", "2020-01-01"]``
          is not considered datetime-parsable, so we won't attempt to convert it).
        - The column as a whole is not datetime-parsable, due to a clash of datetime
          format, e.g. '2020/01/01' and '2020-01-01'.

        When set to ``'coerce'``, the entries of ``X_col`` that should have raised
        an error are set to ``NaT`` instead.
        You can choose which format to use with the keyword argument ``format``, as with
        ``pd.to_datetime``, e.g. ``to_datetime(X_col, format='%Y/%m/%d')``.
        Combined with ``error='coerce'``, this will convert all entries that don't
        match this format to ``NaT``.

        Note that the ``'ignore'`` option is not used and will raise an error.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for the subsampling during
        datetime guessing. Use an int to make the randomness deterministic.

    **kwargs : key, value mappings
        Other keyword arguments are passed down to :func:`pandas.to_datetime`.

        One notable argument is 'format'. Setting a format overwrites
        the datetime format guessing behavior of this function for all columns.

        Note that we don't encourage you to use dayfirst or monthfirst argument, since
        their behavior is ambiguous and might not be applied at all.

        Moreover, this function raises an error if 'unit' is set to any value.
        This is because, in ``pandas.to_datetime``, 'unit' is specific to timestamps,
        whereas in ``skrub.to_datetime`` we don't attempt to parse numeric columns.

    Returns
    -------
    datetime
        Return type depends on input.
        - dataframes, series and 2d arrays return the same type
        - otherwise return the same output as :func:`pandas.to_datetime`.

    See Also
    --------
    :func:`pandas.to_datetime`
        Convert argument to datetime.

    Examples
    --------
    >>> X = pd.DataFrame(dict(a=[1, 2], b=["2021-01-01", "2021-02-02"]))
    >>> X
       a          b
    0  1 2021-01-01
    1  2 2021-02-02
    >>> to_datetime(X)
       a          b
    0  1 2021-01-01
    1  2 2021-02-02
    """
    errors_options = ["coerce", "raise"]
    if errors not in errors_options:
        raise ValueError(f"errors options are {errors_options!r}, got {errors!r}.")
    kwargs["errors"] = errors
    kwargs["random_state"] = random_state

    if "unit" in kwargs:
        raise ValueError(
            "'unit' is not a parameter of skrub.to_datetime; it is only meaningful "
            "when applying pandas.to_datetime to a numerical column"
        )

    # dataframe
    if hasattr(X, "__dataframe__"):
        return _to_datetime_dataframe(X, **kwargs)

    # series, this attribute is available since Pandas 2.1.0
    elif hasattr(X, "__column_consortium_standard__"):
        return _to_datetime_series(X, **kwargs)

    # 2d array
    elif isinstance(X, Iterable) and np.asarray(X).ndim == 2:
        X = _to_datetime_2d_array(np.asarray(X), **kwargs)
        return np.vstack(X).T

    # 1d array
    elif isinstance(X, Iterable) and np.asarray(X).ndim == 1:
        return _to_datetime_1d_array(np.asarray(X), **kwargs)

    # scalar or unknown type
    elif np.asarray(X).ndim == 0:
        return _to_datetime_scalar(X, **kwargs)

    else:
        raise TypeError(
            "X must be a Dataframe, series, 2d array or any "
            f"valid input for ``pd.to_datetime``. Got {X=!r}."
        )


def _to_datetime_dataframe(X, **kwargs):
    """Dataframe specialization of ``_to_datetime_2d``.

    Parameters
    ----------
    X : Pandas or Polars dataframe

    Returns
    -------
    X : Pandas or Polars dataframe
    """
    skrub_px, _ = get_df_namespace(X)
    index = getattr(X, "index", None)

    X_split = [X[col].to_numpy() for col in X.columns]
    X_split = _to_datetime_2d(X_split, **kwargs)

    # TODO: Temporary work-around. Maps back the original non-converted dtypes.
    # Remove this when removing the 'to_numpy()' conversion above.
    datetime_indices, _ = _get_datetime_column_indices(X_split)
    non_datetime_indices = list(set(range(len(X_split))).difference(datetime_indices))
    non_datetime_columns = np.asarray(X.columns)[non_datetime_indices]
    non_datetime_dtypes = np.asarray(X.dtypes)[non_datetime_indices]
    name_to_dtype = dict(zip(non_datetime_columns, non_datetime_dtypes))

    X_split = {col: X_split[col_idx] for col_idx, col in enumerate(X.columns)}

    return skrub_px.make_dataframe(X_split, index=index, dtypes=name_to_dtype)


def _to_datetime_series(X, **kwargs):
    """Series specialization of :func:`pandas.to_datetime`.

    Parameters
    ----------
    X : Pandas or Polars series

    Returns
    -------
    X : Pandas or Polars series
    """
    skrub_px, _ = get_df_namespace(X.to_frame())
    index = getattr(X, "index", None)
    name = X.name
    X_split = [X.to_numpy()]
    X_split = _to_datetime_2d(X_split, **kwargs)

    # TODO: Temporary work-around. Maps back the original non-converted dtype.
    # Remove this when removing the 'to_numpy()' conversion above.
    datetime_indices, _ = _get_datetime_column_indices(X_split)
    if len(datetime_indices) == 1:  # either 1 or 0
        dtype = None
    else:
        dtype = X.dtype

    return skrub_px.make_series(X_split[0], index=index, name=name, dtype=dtype)


def _to_datetime_2d_array(X, **kwargs):
    """2d array specialization of ``_to_datetime_2d``.

    Parameters
    ----------
    X : ndarray of shape ``(n_samples, n_features)``

    Returns
    -------
    X_split : list of array, of shape ``n_features``
    """
    X_split = list(X.T)
    return _to_datetime_2d(X_split, **kwargs)


def _to_datetime_1d_array(X, **kwargs):
    X_split = [X]
    X_split = _to_datetime_2d(X_split, **kwargs)
    return np.asarray(X_split[0])


def _to_datetime_scalar(X, **kwargs):
    X_split = [np.atleast_1d(X)]
    X_split = _to_datetime_2d(X_split, **kwargs)
    return X_split[0][0]


def _to_datetime_2d(
    X_split,
    indices=None,
    index_to_format=None,
    format=None,
    random_state=None,
    **kwargs,
):
    """Convert datetime parsable columns from a 2d array or dataframe \
        to datetime format.

    The conversion is done inplace.

    Parameters
    ----------
    X_split : list of 1d array of length n_features
        The 2d input, chunked into a list of array. This format allows us
        to treat each column individually and preserve their dtype, because
        dataframe.to_numpy() casts all columns to object when at least one
        column dtype is object.

    indices : list of int, default=None
        Indices of the parsable columns to convert.
        If None, indices are computed using the current input X.

    index_to_format : mapping of int to str, default=None
        Dictionary mapping column indices to their datetime format.
        It defines the format parameter for each column when calling
        pd.to_datetime.

        If indices is None, ``indices_to_format`` is computed using the
        current input X.
        If format is not None, all values of ``indices_to_format`` are set
        to format.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for the subsampling during
        datetime guessing. Use an int to make the randomness deterministic.

    format : str, default=None
        When format is not None, it overwrites the values in indices_to_format.

    Returns
    -------
    X_split : list of 1d array of length n_features
    """
    if indices is None:
        indices, index_to_format = _get_datetime_column_indices(X_split, random_state)

    # format overwrite indices_to_format
    if format is not None:
        index_to_format = {col_idx: format for col_idx in indices}

    for col_idx in indices:
        X_split[col_idx] = pd.to_datetime(
            X_split[col_idx], format=index_to_format[col_idx], **kwargs
        )

    return X_split


def _get_datetime_column_indices(X_split, random_state=None):
    """Select the datetime parsable columns by their indices \
    and return their datetime format.

    Parameters
    ----------
    X_split : list of 1d array of length n_features

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for the subsampling during
        datetime guessing. Use an int to make the randomness deterministic.

    Returns
    -------
    datetime_indices : list of int
        List of parsable column, identified by their indices.

    index_to_format: mapping of int to str
        Dictionary mapping parsable column indices to their datetime format.
    """
    indices = []
    index_to_format = {}

    for col_idx, X_col in enumerate(X_split):
        X_col = X_col[pd.notnull(X_col)]  # X_col is a numpy array

        if is_numeric_dtype(X_col):
            continue

        elif is_datetime64_any_dtype(X_col):
            indices.append(col_idx)
            index_to_format[col_idx] = None

        elif _is_column_datetime_parsable(X_col):
            indices.append(col_idx)

            # _guess_datetime_format only accept string columns.
            # We need to filter out columns of object dtype that
            # contains e.g., datetime.datetime or pd.Timestamp.
            X_col_str = X_col.astype(str)
            if np.array_equal(X_col, X_col_str):
                datetime_format = _guess_datetime_format(X_col, random_state)
            else:
                # We don't need to specify a parsing format
                # for columns that are already of type datetime64.
                datetime_format = None

            index_to_format[col_idx] = datetime_format

    return indices, index_to_format


def _is_column_datetime_parsable(X_col):
    """Check whether a 1d array can be converted into a \
    :class:`pandas.DatetimeIndex`.

    Parameters
    ----------
    X_col : array-like of shape ``(n_samples,)``, of dtype str or object.

    Returns
    -------
    is_dt_parsable : bool
    """
    # Remove columns of int, float or bool casted as object.
    # Pandas < 2.0.0 raise a deprecation warning instead of an error.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        try:
            if np.array_equal(X_col, X_col.astype(np.float64)):
                return False
        except (ValueError, TypeError):
            pass

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            # format=mixed parses entries individually,
            # avoiding ValueError when both date and datetime formats
            # are present.
            # At this stage, the format itself doesn't matter.
            _ = pd.to_datetime(X_col, format=MIXED_FORMAT)
        return True
    except (pd.errors.ParserError, ValueError, TypeError):
        return False


def _guess_datetime_format(X_col, random_state=None):
    """Infer the format of a 1d array.

    This functions uses Pandas ``guess_datetime_format`` routine for both
    dayfirst and monthfirst case, and select either format when using one
    give a unify format on the array.

    When both dayfirst and monthfirst format are possible, we select
    monthfirst by default.

    You can overwrite this behaviour by setting a format in the caller function.
    Setting a format always take precedence over infering it using
    ``_guess_datetime_format``.

    For computational effiency, we only subsample ``n_samples`` rows of ``X_col``.
    ``n_samples`` is currently set to 30.

    Parameters
    ----------
    X_col : ndarray of shape ``(n_samples,)``
        X_col must only contains string objects without any missing value.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for the subsampling during
        datetime guessing. Use an int to make the randomness deterministic.

    Returns
    -------
    datetime_format : str or None
    """
    # Passing numpy.str_ (i.e. dtype '<U10') to 'guess_datetime_format'
    # raises a TypeError.
    # We have to convert these to the object dtype first.
    X_col = X_col.astype("object")

    # Subsample samples for fast format estimation
    n_samples = 30
    size = min(X_col.shape[0], n_samples)
    rng = check_random_state(random_state)
    X_col = rng.choice(X_col, size=size, replace=False)

    vfunc = np.vectorize(guess_datetime_format)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        # pd.unique handles None
        month_first_formats = pd.unique(vfunc(X_col, dayfirst=False))
        day_first_formats = pd.unique(vfunc(X_col, dayfirst=True))

    if len(month_first_formats) == 1 and month_first_formats[0] is not None:
        return str(month_first_formats[0])

    elif len(day_first_formats) == 1 and day_first_formats[0] is not None:
        return str(day_first_formats[0])

    # special heuristic: when both date and datetime formats are
    # present, allow the format to be mixed.
    elif (
        len(month_first_formats) == 2
        and len(day_first_formats) == 2
        and len(month_first_formats[0]) != len(month_first_formats[1])
    ):
        return MIXED_FORMAT

    else:
        return None


def _is_column_date_only(X_col):
    """Check whether a :obj:`pandas.DatetimeIndex` only contains dates.

    Parameters
    ----------
    X_col : pandas.DatetimeIndex of shape ``(n_samples,)``

    Returns
    -------
    is_date : bool
    """
    return np.array_equal(X_col, X_col.normalize())


def _datetime_to_total_seconds(X_col):
    """
    Parameters
    ----------
    X_col : DatetimeIndex of shape (n_samples,)

    Returns
    -------
    X_col : ndarray of shape (n_samples)
    """
    if X_col.tz is not None:
        X_col = X_col.tz_convert("utc")

    # Total seconds since epoch
    mask_notnull = X_col == X_col

    return np.where(
        mask_notnull,
        X_col.astype("int64") / 1e9,
        np.nan,
    )


class DatetimeEncoder(TransformerMixin, BaseEstimator):
    """Transforms each datetime column into several numeric columns \
    for temporal features (e.g year, month, day...).

    If the dates are timezone aware, all the features extracted will correspond
    to the provided timezone.

    Parameters
    ----------
    resolution : {"year", "month", "day", "hour", "minute", "second",
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

    errors : {'coerce', 'raise'}, default="coerce"
        During transform:
        - If ``"coerce"``, then invalid parsing will be set as ``pd.NaT``.
        - If ``"raise"``, then invalid parsing will raise an exception.

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
        errors="coerce",
    ):
        self.resolution = resolution
        self.add_day_of_the_week = add_day_of_the_week
        self.add_total_seconds = add_total_seconds
        self.errors = errors

    def fit(self, X, y=None):
        """Fit the instance to X.

        Select datetime-parsable columns and generate the list of
        datetime feature to extract.

        Parameters
        ----------
        X : array-like, shape ``(n_samples, n_features)``
            Input data. Columns that can't be converted into
            ``pandas.DatetimeIndex`` and numerical values will
            be dropped.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        DatetimeEncoder
            Fitted DatetimeEncoder instance (self).
        """
        if self.resolution not in TIME_LEVELS and self.resolution is not None:
            raise ValueError(
                f"'resolution' options are {TIME_LEVELS}, got {self.resolution!r}."
            )

        errors_options = ["coerce", "raise"]
        if self.errors not in errors_options:
            raise ValueError(
                f"'errors' options are {errors_options!r}, got {self.errors!r}."
            )

        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)
        X = check_array(
            X, ensure_2d=True, force_all_finite=False, dtype=None, copy=False
        )

        self._select_datetime_cols(X)

        return self

    def _select_datetime_cols(self, X):
        """Select datetime-parsable columns and generate the list of
        datetime feature to extract.

        If the input only contains dates (and no datetimes), only the features
        ["year", "month", "day"] will be filtered with resolution.

        Parameters
        ----------
        X : array-like of shape ``(n_samples, n_features)``
        """
        if self.resolution is None:
            levels = []
        else:
            idx_level = TIME_LEVELS.index(self.resolution)
            levels = TIME_LEVELS[: idx_level + 1]

        X_split = np.hsplit(X, X.shape[1])
        self.column_indices_, self.index_to_format_ = _get_datetime_column_indices(
            X_split
        )

        self.index_to_features_ = defaultdict(list)
        self.n_features_out_ = 0

        for col_idx in self.column_indices_:
            X_col = pd.DatetimeIndex(X[:, col_idx])
            if _is_column_date_only(X_col):
                # Keep only date attributes
                levels = [
                    level for level in levels if level in ["year", "month", "day"]
                ]

            self.index_to_features_[col_idx] += levels
            self.n_features_out_ += len(levels)

            if self.add_total_seconds:
                self.index_to_features_[col_idx].append("total_seconds")
                self.n_features_out_ += 1

            if self.add_day_of_the_week:
                self.index_to_features_[col_idx].append("day_of_week")
                self.n_features_out_ += 1

    def transform(self, X, y=None):
        """Transform ``X`` by replacing each datetime column with \
        corresponding numerical features.

        Parameters
        ----------
        X : array-like of shape ``(n_samples, n_features)``
            The data to transform, where each column is a datetime feature.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        X_out : ndarray of shape ``(n_samples, n_features_out_)``
            Transformed input.
        """
        check_is_fitted(self)
        self._check_n_features(X, reset=False)
        self._check_feature_names(X, reset=False)

        X = check_array(
            X,
            ensure_2d=True,
            force_all_finite=False,
            dtype=None,
            copy=False,
        )
        X_split = _to_datetime_2d_array(
            X,
            indices=self.column_indices_,
            index_to_format=self.index_to_format_,
            errors=self.errors,
        )

        return self._extract_features(X_split)

    def _extract_features(self, X_split):
        """Extract datetime features from the selected columns.

        Parameters
        ----------
        X_split : list of 1d array of length n_features

        Returns
        -------
        X_out : ndarray of shape ``(n_samples, n_features_out_)``
        """
        # X_out must be of dtype float64 otherwise np.nan will overflow
        # to large negative numbers.
        X_out = np.empty((X_split[0].shape[0], self.n_features_out_), dtype=np.float64)
        offset_idx = 0
        for col_idx in self.column_indices_:
            X_col = X_split[col_idx]
            features = self.index_to_features_[col_idx]
            for feat_idx, feature in enumerate(features):
                if feature == "total_seconds":
                    X_feature = _datetime_to_total_seconds(X_col)
                else:
                    X_feature = getattr(X_col, feature).to_numpy()
                X_out[:, offset_idx + feat_idx] = X_feature

            offset_idx += len(features)

        return X_out

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation.

        Feature names are formatted like: "<column_name>_<new_feature>"
        if the original data has column names, otherwise with format
        "<column_index>_<new_feature>" where `<new_feature>` is one of
        {"year", "month", "day", "hour", "minute", "second",
        "microsecond", "nanosecond", "day_of_week"}.

        Parameters
        ----------
        input_features : None
            Unused, only here for compatibility.

        Returns
        -------
        feature_names : ndarray of str
            List of feature names.
        """
        check_is_fitted(self, "index_to_features_")
        feature_names = []
        columns = getattr(self, "feature_names_in_", list(range(self.n_features_in_)))
        for col_idx, features in self.index_to_features_.items():
            column = columns[col_idx]
            feature_names += [f"{column}_{feat}" for feat in features]
        return np.asarray(feature_names, dtype=object)

    def _more_tags(self):
        """
        Used internally by sklearn to ease the estimator checks.
        """
        return {
            "X_types": ["2darray", "categorical"],
            "allow_nan": True,
            "_xfail_checks": {"check_dtype_object": "Specific datetime error."},
        }
