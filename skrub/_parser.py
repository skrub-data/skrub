import warnings

import numpy as np
import pandas as pd
from pandas._libs.tslibs.parsing import guess_datetime_format
from pandas.api.types import CategoricalDtype, is_datetime64_any_dtype, is_numeric_dtype
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.fixes import parse_version
from sklearn.utils.validation import check_random_state

from ._dataframe._namespace import get_df_namespace


def _is_pandas_format_mixed_available():
    pandas_version = pd.__version__
    min_pandas_version = "2.0.0"
    return parse_version(min_pandas_version) < parse_version(pandas_version)


MIXED_FORMAT = "mixed" if _is_pandas_format_mixed_available() else None


class _BaseParser(TransformerMixin, BaseEstimator):
    """Base class to define parsers on dataframes.

    This class is a helper for type parsing operations, used
    in TableVectorizer and DatetimeEncoder. The goal of parsing is to
    apply the columns types seen during fit automatically during transform,
    and improve the downstream analytics or learning task.

    During fit, each columns are parsed against a specific dtype
    matching a subclass implementation, and the mapping between columns
    and inferred dtype are saved in inferred_column_types_.
    During transform, columns are casted to the dtype seen during fit.

    Subclasses of this estimator overwrite _infer and _parse methods.

    Parameters
    ----------
    errors : {'coerce', 'raise'}, default='coerce'
        During transform:
        - If 'coerce', then invalid column values will be set as ``pd.NaT``
        or ``np.nan``, depending on the parser.
        - If 'raise', then invalid parsing will raise an exception.

    Attributes
    ----------
    inferred_column_types_ : mapping of a string to dtype
        The saved infered dtypes.
    """

    def __init__(self, errors="coerce"):
        self.errors = errors

    def fit(self, X, y=None):
        """Parse X and save its inferred dtypes.

        Parameters
        ----------
        X : {polars, pandas}.DataFrame of shape (n_samples, n_features).
            The input is converted into a Pandas dataframe.
        y : None
            Unused, here for compatibility with scikit-learn.

        Returns
        -------
        _Parser : self
            A fitted instance of a subclass.
        """
        del y

        if not hasattr(X, "__dataframe__"):
            raise TypeError(f"X must be a dataframe, got {type(X)}")

        # TODO: remove this line and enable Polars operations.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self._check_feature_names(X, reset=True)
        self._check_n_features(X, reset=True)

        self.inferred_column_types_ = dict()
        for col in X.columns:
            dtype = self._infer(col, X[col])
            if dtype is not None:
                self.inferred_column_types_[col] = dtype

        return self

    def transform(self, X, y=None):
        """Cast X columns using the dtypes parsed during fit.

        These operations are performed on a copy of X.

        Parameters
        ----------
        X : {pandas, polars}.DataFrame of shape (n_samples, n_features)
            The input is converted into a Pandas dataframe.
        y : None
            Unused, here for compatibility with scikit-learn.

        Returns
        -------
        X : {pandas, polars}.DataFrame of shape (n_samples, n_features)
            A copy of the input with columns casted to the dtypes seen during fit.
        """
        del y

        if not hasattr(X, "__dataframe__"):
            raise TypeError(f"X must be a dataframe, got {type(X)}")

        skrub_px, _ = get_df_namespace(X)
        index = getattr(X, "index", None)

        # TODO: remove this line and enable Polars operations.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self._check_feature_names(X, reset=False)
        self._check_n_features(X, reset=False)

        X_out = dict()
        for col in X.columns:
            if col in self.inferred_column_types_:
                X_out[col] = self._parse(col, X[col])
            else:
                X_out[col] = X[col]

        return skrub_px.make_dataframe(X_out, index=index)

    def _infer(self, column_name, column):
        """Infer the dtype of a column.

        This method is overwritten in subclasses.

        Parameters
        ----------
        column_name : str
            The name of the input column.
        column : {pandas, polars}.Series
            The column whose dtype is inferred against a specific dtype.

        Returns
        -------
        dtype : dtype or None
            The inferred dtype. The output is None if the column couldn't
            be parsed.
        """
        raise NotImplementedError()

    def _parse(self, column_name, column):
        """Parse a column against its dtype seen during _infer.

        This method is overwritten in subclasses.

        Parameters
        ----------
        column_name : str
            The name of the input column.
        column : {pandas, polars}.Series
            The input column to be parsed.

        Returns
        -------
        column : {pandas, polars}.Series
            The input column converted to the dtype seen during _infer.
        """
        raise NotImplementedError()


def _is_column_datetime_parsable(column):
    """Check whether a 1d array can be converted into a \
    :class:`pandas.DatetimeIndex`.

    Parameters
    ----------
    column : array-like of shape ``(n_samples,)``

    Returns
    -------
    is_dt_parsable : bool
    """
    # The aim of this section is to remove any columns of int, float or bool
    # casted as object.
    # Pandas < 2.0.0 raise a deprecation warning instead of an error.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        try:
            if np.array_equal(column, column.astype(np.float64)):
                return False
        except (ValueError, TypeError):
            pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            # format=mixed parses entries individually,
            # avoiding ValueError when both date and datetime formats
            # are present.
            # At this stage, the format itself doesn't matter.
            _ = pd.to_datetime(column, format=MIXED_FORMAT)
            return True
        except (pd.errors.ParserError, ValueError, TypeError):
            return False


def _guess_datetime_format(column, random_state=None):
    """Infer the format of a 1d array.

    This functions uses Pandas ``guess_datetime_format`` routine for both
    dayfirst and monthfirst case, and select either format when using one
    give a unify format on the array.

    When both dayfirst and monthfirst format are possible, we select
    monthfirst by default.

    You can overwrite this behaviour by setting the format argument of the
    caller function.
    Setting a format always take precedence over infering it using
    ``_guess_datetime_format``.
    """
    # Subsample samples for fast format estimation
    n_samples = 30
    size = min(column.shape[0], n_samples)
    rng = check_random_state(random_state)
    column = rng.choice(column, size=size, replace=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)

        # In Pandas 2.1, pd.to_datetime has a different behavior
        # when the items of column are np.str_ instead of str.
        # TODO: find the bug and remove this line
        column = pd.Series(map(str, column))

        # pd.unique handles None
        month_first_formats = column.apply(
            guess_datetime_format, dayfirst=False
        ).unique()
        day_first_formats = column.apply(guess_datetime_format, dayfirst=True).unique()

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


class _DatetimeParser(_BaseParser):
    """Parse datetime columns of a Pandas or Polars dataframe.

    Parameters
    ----------
    See the docstring of skrub._datetime_encoder.to_datetime.

    Attributes
    ----------
    inferred_column_formats_ : mapping of string to string or DateTime64DType.
        Mapping between column names infered  and their format.
        If the column is already of datetime dtype, we use this as a format.
    """

    def __init__(
        self, format=None, errors="coerce", random_state=None, **to_datetime_params
    ):
        super().__init__(errors=errors)
        self.format = format
        self.random_state = random_state
        self.to_datetime_params = to_datetime_params

    def fit(self, X, y=None):
        del y
        self.inferred_column_formats_ = dict()
        return super().fit(X)

    def _infer(self, column_name, column):
        column = column.to_numpy()  # TODO: remove
        column = column[pd.notnull(column)]

        if is_numeric_dtype(column):
            return None

        elif is_datetime64_any_dtype(column):
            self.inferred_column_formats_[column_name] = None
            return column.dtype

        elif _is_column_datetime_parsable(column):
            # _guess_datetime_format only accept string columns.
            # We need to filter out columns of object dtype that
            # contains e.g., datetime.datetime or pd.Timestamp.
            column_str = column.astype(str)
            if np.array_equal(column, column_str):
                datetime_format = _guess_datetime_format(column, self.random_state)
            else:
                # We don't need to specify a parsing format
                # for columns that are already of type datetime64.
                datetime_format = None
            self.inferred_column_formats_[column_name] = datetime_format
            return column.dtype

        return None

    def _parse(self, column_name, column):
        # In Pandas 2.1, pd.to_datetime has a different behavior
        # when the items of column are np.str_ instead of str.
        # TODO: find the bug and remove this line
        column = list(map(str, column))
        datetime_format = self.format or self.inferred_column_formats_[column_name]
        return pd.to_datetime(
            column,
            format=datetime_format,
            errors=self.errors,
            **self.to_datetime_params,
        )


class _NumericParser(_BaseParser):
    """Parse numeric columns of a Pandas dataframe."""

    def _infer(self, column_name, column):
        if is_datetime64_any_dtype(column):
            return None

        try:
            return pd.to_numeric(column, errors="raise").dtype
        except (ValueError, TypeError):
            return None

    def _parse(self, column_name, column):
        return pd.to_numeric(column, errors=self.errors)


def _union_category(column, dtype):
    """Update a categorical dtype with new entries.

    Parameters
    ----------
    column : pandas.Series
        The input data.

    dtype : CategoricalDtype
        The categorical dtype to update using the values from the column.

    Returns
    -------
    dtype : CategoricalDtype
        The updated categorical dtype.
    """
    known_categories = dtype.categories
    column_categories = pd.unique(column.loc[column.notnull()])
    updated_categories = known_categories.union(column_categories)
    dtype = pd.CategoricalDtype(categories=updated_categories)
    return dtype


class _CategoryParser(_BaseParser):
    """Parse string, object and categorical columns of a Pandas dataframe."""

    def _infer(self, column_name, column):
        if is_numeric_dtype(column) or is_datetime64_any_dtype(column):
            return None
        return column.dtype

    def _parse(self, column_name, column):
        dtype = self.inferred_column_types_[column_name]
        # astype doesn't support errors == "coerce"
        astype_errors = "ignore" if self.errors == "coerce" else "raise"
        column = column.astype(dtype, errors=astype_errors)

        if isinstance(column.dtype, CategoricalDtype):
            dtype = _union_category(column, dtype)
            self.inferred_column_types_[column_name] = dtype
            column = column.astype(dtype)

        return column
