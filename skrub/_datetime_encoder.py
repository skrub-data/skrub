import warnings
from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from pandas._libs.tslibs.parsing import guess_datetime_format
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.fixes import parse_version
from sklearn.utils.validation import check_is_fitted

from .dataframe._namespace import get_df_namespace

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
    **kwargs,
):
    """
    Convert argument to datetime.

    Augment :func:`pandas.to_datetime` by supporting dataframes
    and 2d arrays inputs. It converts compatible columns to datetime, and
    pass incompatible columns unchanged.

    int, float, str, datetime, list, tuple, 1d array, and Series are defered to
    :func:`pandas.to_datetime` directly.

    Parameters
    ----------
    X : int, float, str, datetime, list, tuple, nd array, Series, DataFrame/dict-like
        The object to convert to a datetime.

    errors : {'ignore', 'raise', 'coerce'}, default 'coerce'
        - If ``'raise'``, then invalid parsing will raise an exception.
        - If ``'coerce'``, then invalid parsing will be set as ``NaT``.
        Note that ``'ignore'`` is not used for dataframes, 2d arrays,
        and series, and is used otherwise as in ``pd.to_datetime``.

    **kwargs : key, value mappings
        Other keyword arguments are passed down to
        :func:`pandas.to_datetime`.

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
    """
    kwargs["errors"] = errors

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

    # scalar or unknown type
    return pd.to_datetime(X, **kwargs)


def _to_datetime_dataframe(X, **kwargs):
    """Dataframe specialization of ``_to_datetime_2d``.

    Parameters
    ----------
    X : Pandas or Polars dataframe

    Returns
    -------
    X : Pandas or Polars dataframe
    """
    _, px = get_df_namespace(X)
    index = getattr(X, "index", None)
    X_split = [X[col].to_numpy() for col in X.columns]
    X_split = _to_datetime_2d(X_split, **kwargs)
    X_split = {col: X_split[col_idx] for col_idx, col in enumerate(X.columns)}
    X = pd.DataFrame(X_split, index=index)
    # conversion is px is Polars, no-op if Pandas
    return px.DataFrame(X)


def _to_datetime_series(X, **kwargs):
    """Series specialization of :func:`pandas.to_datetime`.

    Parameters
    ----------
    X : Pandas or Polars series

    Returns
    -------
    X : Pandas or Polars series
    """
    _, px = get_df_namespace(X.to_frame())
    index = getattr(X, "index", None)
    name = X.name
    X_split = [X.to_numpy()]
    X_split = _to_datetime_2d(X_split)
    X = pd.Series(X_split[0], index=index, name=name)
    # conversion is px is Polars, no-op if Pandas
    return px.Series(X)


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


def _to_datetime_2d(
    X_split,
    indices=None,
    index_to_format=None,
    format=None,
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

        If indices is None, indices_to_format is computed using the current input X.
        If format is not None, all values of indices_to_format are format.

    format : str, default=None
        Here for compatibility with ``pandas.to_datetime`` API.
        When format is not None, it overwrites the values in indices_to_format.

    Returns
    -------
    X_split : list of 1d array of length n_features
    """
    if indices is None:
        indices, index_to_format = _get_datetime_column_indices(X_split)

    # format overwrite indices_to_format
    if format is not None:
        index_to_format = {col_idx: format for col_idx in indices}

    for col_idx in indices:
        X_split[col_idx] = pd.to_datetime(
            X_split[col_idx], format=index_to_format[col_idx], **kwargs
        )

    return X_split


def _get_datetime_column_indices(X_split):
    """Select the datetime parsable columns by their indices \
    and return their datetime format.

    Parameters
    ----------
    X_split : list of 1d array of length n_features

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
        X_col = X_col[pd.notnull(X_col)]

        # convert pd.TimeStamp to np.datetime64
        if all(isinstance(val, pd.Timestamp) for val in X_col):
            X_col = X_col.astype("datetime64")

        if _is_column_datetime_parsable(X_col):
            indices.append(col_idx)
            # TODO: pass require_dayfirst to _guess_datetime_format
            index_to_format[col_idx] = _guess_datetime_format(X_col)

    return indices, index_to_format


def _is_column_datetime_parsable(X_col):
    """Check whether a 1d array can be converted into a \
    :class:`pandas.DatetimeIndex`.

    Parameters
    ----------
    X_col : array-like of shape ``(n_samples,)``

    Returns
    -------
    is_dt_parsable : bool
    """
    # Remove columns of int, float or bool casted as object.
    try:
        if np.array_equal(X_col, X_col.astype(np.float64)):
            return False
    except ValueError:
        pass

    np_dtypes_candidates = [np.object_, np.str_, np.datetime64]
    is_type_datetime_compatible = any(
        np.issubdtype(X_col.dtype, np_dtype) for np_dtype in np_dtypes_candidates
    )
    if is_type_datetime_compatible:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                # format=mixed parses entries individually,
                # avoiding ValueError when both date and datetime formats
                # are present.
                # At this stage, the format itself doesn't matter.
                _ = pd.to_datetime(X_col, format=MIXED_FORMAT)
            return True
        except (pd.errors.ParserError, ValueError):
            pass
    return False


def _guess_datetime_format(X_col, require_dayfirst=False):
    """
    Parameters
    ----------
    X_col : ndarray of shape ``(n_samples,)``

    require_dayfirst : bool, default True
        Whether to return the dayfirst format when both dayfirst
        and monthfirst are valid.

    Returns
    -------
    format : str
    """
    if np.issubdtype(X_col.dtype, np.datetime64):
        # We don't need to specify a parsing format
        # for columns that are already of type datetime64.
        return None

    X_col = X_col.astype(np.object_)
    vfunc = np.vectorize(guess_datetime_format)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        # pd.unique handles None
        month_first_formats = pd.unique(vfunc(X_col, dayfirst=False))
        day_first_formats = pd.unique(vfunc(X_col, dayfirst=True))

    if None in month_first_formats or None in day_first_formats:
        return None

    elif (
        len(month_first_formats) == 1
        and len(day_first_formats) == 1
        and month_first_formats[0] != day_first_formats[0]
    ):
        if require_dayfirst:
            return str(day_first_formats[0])
        else:
            return str(month_first_formats[0])

    elif len(month_first_formats) == 1:
        return str(month_first_formats[0])

    elif len(day_first_formats) == 1:
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
        If ``None``, no feature will be created.

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
            `pandas.DatetimeIndex` and numerical values will
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
                f"errors options are {errors_options!r}, got {self.errors!r}."
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
        del X_split

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
        feature_names : list of str
            List of feature names.
        """
        check_is_fitted(self, "index_to_features_")
        feature_names = []
        columns = getattr(self, "feature_names_in_", list(range(self.n_features_in_)))
        for col_idx, features in self.index_to_features_.items():
            column = columns[col_idx]
            feature_names += [f"{column}_{feat}" for feat in features]
        return feature_names

    def _more_tags(self):
        """
        Used internally by sklearn to ease the estimator checks.
        """
        return {
            "X_types": ["2darray", "categorical"],
            "allow_nan": True,
            "_xfail_checks": {"check_dtype_object": "Specific datetime error."},
        }
