from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from ._parser import _DatetimeParser

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
        When set to 'raise', errors will be raised only when all
        the following conditions are satisfied, for a given column:
        - After converting to numpy, the column dtype is np.object_ or np.str_
        - Each entry of the column is datetime-parsable, i.e.
          ``pd.to_datetime(X_col, format='mixed')`` doesn't raise an error.
          This step is conservative, because e.g.
          ``["2020-01-01", "hello", "2020-01-01"]``
          is not considered datetime-parsable, so we won't attempt to convert it.
        - The column as a whole is not datetime-parsable, due to a clash of datetime
          format, e.g. '2020/01/01' and '2020-01-01'.

        When set to 'coerce', the entries of the column that should have raised
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
    """Convert the columns of a Pandas or Polars dataframe into \
        datetime representation.

    Parameters
    ----------
    X : Pandas or Polars dataframe

    Returns
    -------
    X : Pandas or Polars dataframe
    """
    datetime_parser = _DatetimeParser(**kwargs)
    return datetime_parser.fit_transform(X)


def _to_datetime_series(X, **kwargs):
    """Convert a Pandas or Polars series into datetime representation.

    Parameters
    ----------
    X : Pandas or Polars series

    Returns
    -------
    X : Pandas or Polars series
    """
    X = X.to_frame()
    datetime_parser = _DatetimeParser(**kwargs)
    X = datetime_parser.fit_transform(X)
    return X[X.columns[0]]


def _to_datetime_2d_array(X, **kwargs):
    """Convert a 2d-array into datetime representation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    """
    X = pd.DataFrame(X)
    datetime_parser = _DatetimeParser(**kwargs)
    X = datetime_parser.fit_transform(X)
    return X.to_numpy()


def _to_datetime_1d_array(X, **kwargs):
    X = pd.DataFrame([X]).T
    datetime_parser = _DatetimeParser(**kwargs)
    X = datetime_parser.fit_transform(X)
    return X[X.columns[0]].to_numpy()


def _to_datetime_scalar(X, **kwargs):
    X = pd.DataFrame([np.atleast_1d(X)])
    datetime_parser = _DatetimeParser(**kwargs)
    X = datetime_parser.fit_transform(X)
    return X[0][0]


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

        # TODO: remove this line and perform dataframes operations only
        # across this class.
        if not hasattr(X, "__dataframe__"):
            X = pd.DataFrame(X)
        self._datetime_parser = _DatetimeParser(errors=self.errors).fit(X)

        X = check_array(
            X, ensure_2d=True, force_all_finite=False, dtype=None, copy=False
        )

        self._select_datetime_features(X)

        return self

    def _select_datetime_features(self, X):
        """Select datetime-parsable columns and generate the list of
        datetime feature to extract.

        If the input only contains dates (and no datetimes), only the features
        ["year", "month", "day"] will be filtered with resolution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        """
        if self.resolution is None:
            levels = []
        else:
            idx_level = TIME_LEVELS.index(self.resolution)
            levels = TIME_LEVELS[: idx_level + 1]

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

        # TODO: remove this line and perform dataframes operations only
        # across this class.
        if not hasattr(X, "__dataframe__"):
            X = pd.DataFrame(X)

        X = self._datetime_parser.transform(X)

        X = check_array(
            X,
            ensure_2d=True,
            force_all_finite=False,
            dtype=None,
            copy=False,
        )

        return self._extract_features(X)

    def _extract_features(self, X):
        """Extract datetime features from the selected columns.

        Parameters
        ----------
        X_split : ndarray of shape (n_samples, n_features)

        Returns
        -------
        X_out : ndarray of shape (n_samples, n_features_out_)
        """
        # X_out must be of dtype float64 otherwise np.nan will overflow
        # to large negative numbers.
        X_out = np.empty((X.shape[0], self.n_features_out_), dtype=np.float64)
        offset_idx = 0
        for col_idx in self.column_indices_:
            X_col = pd.DatetimeIndex(X[:, col_idx])
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

    @property
    def column_indices_(self):
        """Indices of the datetime-parsable columns."""
        datetime_cols = list(self._datetime_parser.inferred_column_types_)
        columns = pd.Series(
            getattr(
                self._datetime_parser,
                "feature_names_in_",
                list(range(self._datetime_parser.n_features_in_)),
            )
        )
        return columns[columns.isin(datetime_cols)].index.tolist()

    @property
    def index_to_format_(self):
        """Mapping from column indices to their datetime formats."""
        formats = self._datetime_parser.inferred_column_formats_
        columns = getattr(
            self._datetime_parser,
            "feature_names_in_",
            list(range(self._datetime_parser.n_features_in_)),
        )
        return {idx: formats[columns[idx]] for idx in self.column_indices_}

    def _more_tags(self):
        """
        Used internally by sklearn to ease the estimator checks.
        """
        return {
            "X_types": ["2darray", "categorical"],
            "allow_nan": True,
            "_xfail_checks": {"check_dtype_object": "Specific datetime error."},
        }
