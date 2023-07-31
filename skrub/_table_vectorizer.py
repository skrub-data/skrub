"""
Implements the TableVectorizer: a preprocessor to automatically apply
transformers/encoders to different types of data, without the need to
manually categorize them beforehand, or construct complex Pipelines.
"""

import re
import warnings
from typing import Literal
from warnings import warn

import numpy as np
import pandas as pd
import sklearn
from numpy.typing import ArrayLike, NDArray
from pandas._libs.tslibs.parsing import guess_datetime_format
from pandas.core.dtypes.base import ExtensionDtype
from scipy import sparse
from sklearn.base import TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.deprecation import deprecated
from sklearn.utils.validation import check_is_fitted

from skrub import DatetimeEncoder, GapEncoder
from skrub._utils import parse_astype_error_message

# Required for ignoring lines too long in the docstrings
# flake8: noqa: E501


def _infer_date_format(date_column: pd.Series, n_trials: int = 100) -> str | None:
    """Infer the date format of a date column,
    by finding a format which should work for all dates in the column.

    Parameters
    ----------
    date_column : Series
        A column of dates, as strings.
    n_trials : int, default=100
        Number of rows to use to infer the date format.

    Returns
    -------
    str or None
        The date format inferred from the column.
        If no format could be inferred, returns None.
    """
    if len(date_column) == 0:
        return
    date_column_sample = date_column.dropna().sample(
        frac=min(n_trials / len(date_column), 1), random_state=42
    )
    # try to infer the date format
    # see if either dayfirst or monthfirst works for all the rows
    with warnings.catch_warnings():
        # pandas warns when dayfirst is not strictly applied
        warnings.simplefilter("ignore")
        date_format_monthfirst = date_column_sample.apply(
            lambda x: guess_datetime_format(x)
        )
        date_format_dayfirst = date_column_sample.apply(
            lambda x: guess_datetime_format(x, dayfirst=True),
        )
    # if one row could not be parsed, return None
    if date_format_monthfirst.isnull().any() or date_format_dayfirst.isnull().any():
        return
    # even with dayfirst=True, monthfirst format can be inferred
    # so we need to check if the format is the same for all the rows
    elif date_format_monthfirst.nunique() == 1:
        # one monthfirst format works for all the rows
        # check if another format works for all the rows
        # if so, raise a warning
        if date_format_dayfirst.nunique() == 1:
            # check if monthfirst and dayfirst haven't found the same format
            if date_format_monthfirst.iloc[0] != date_format_dayfirst.iloc[0]:
                warnings.warn(
                    f"""
                    Both {date_format_monthfirst.iloc[0]} and {date_format_dayfirst.iloc[0]} are valid
                    formats for the dates in column '{date_column.name}'.
                    Format {date_format_monthfirst.iloc[0]} will be used.
                    """,
                    UserWarning,
                    stacklevel=2,
                )
        return date_format_monthfirst.iloc[0]
    elif date_format_dayfirst.nunique() == 1:
        # only this format works for all the rows
        return date_format_dayfirst.iloc[0]
    else:
        # more than two different formats were found
        # TODO: maybe we could deal with this case
        return


def _has_missing_values(df: pd.DataFrame | pd.Series) -> bool:
    """
    Returns True if `array` contains missing values, False otherwise.
    """
    return any(df.isnull())


def _replace_false_missing(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Takes a DataFrame or a Series, and replaces the "false missing", that is,
    strings that designate a missing value, but do not have the corresponding
    type. We convert these strings to np.nan.
    Also replaces `None` to np.nan.
    """
    # Should not replace "missing" (the string used for imputation in
    # categorical features).
    STR_NA_VALUES = [
        "null",
        "",
        "1.#QNAN",
        "#NA",
        "nan",
        "#N/A N/A",
        "-1.#QNAN",
        "<NA>",
        "-1.#IND",
        "-nan",
        "n/a",
        "-NaN",
        "1.#IND",
        "NULL",
        "NA",
        "N/A",
        "#N/A",
        "NaN",
    ]  # taken from pandas.io.parsers (version 1.1.4)
    df = df.replace(STR_NA_VALUES + [None, "?", "..."], np.nan)
    df = df.replace(r"^\s+$", np.nan, regex=True)  # Replace whitespaces
    return df


def _replace_missing_in_cat_col(ser: pd.Series, value: str = "missing") -> pd.Series:
    """
    Takes a Series with string data,
    replaces the missing values, and returns it.
    """
    ser = _replace_false_missing(ser)
    if isinstance(ser.dtype, pd.CategoricalDtype) and (value not in ser.cat.categories):
        ser = ser.cat.add_categories([value])
    ser = ser.fillna(value=value)
    return ser


OptionalTransformer = (
    TransformerMixin | Literal["drop", "remainder", "passthrough"] | None
)


class TableVectorizer(ColumnTransformer):
    """Automatically transform a heterogeneous dataframe to a numerical array.

    Easily transforms a heterogeneous data table
    (such as a :obj:`~pandas.DataFrame`) to a numerical array for machine
    learning. For this it transforms each column depending on its data type.
    It provides a simplified interface for the ColumnTransformer ;
    more documentation of attributes and functions are available in its doc.

    .. versionadded:: 0.2.0

    Parameters
    ----------
    cardinality_threshold : int, default=40
        Two lists of features will be created depending on this value: strictly
        under this value, the low cardinality categorical features, and above or
        equal, the high cardinality categorical features.
        Different transformers will be applied to these two groups,
        defined by the parameters `low_card_cat_transformer` and
        `high_card_cat_transformer` respectively.
        Note: currently, missing values are counted as a single unique value
        (so they count in the cardinality).

    low_card_cat_transformer : {'drop', 'remainder', 'passthrough'} or Transformer, optional
        Transformer used on categorical/string features with low cardinality
        (threshold is defined by `cardinality_threshold`).
        Can either be a transformer object instance (e.g. OneHotEncoder),
        a Pipeline containing the preprocessing steps,
        'drop' for dropping the columns,
        'remainder' for applying `remainder`,
        'passthrough' to return the unencoded columns,
        or `None` to use the default transformer
        (OneHotEncoder(handle_unknown="ignore", drop="if_binary")).
        Features classified under this category are imputed based on the
        strategy defined with `impute_missing`.

    high_card_cat_transformer : {'drop', 'remainder', 'passthrough'} or Transformer, optional
        Transformer used on categorical/string features with high cardinality
        (threshold is defined by `cardinality_threshold`).
        Can either be a transformer object instance
        (e.g. GapEncoder), a Pipeline containing the preprocessing steps,
        'drop' for dropping the columns,
        'remainder' for applying `remainder`,
        'passthrough' to return the unencoded columns,
        or `None` to use the default transformer (GapEncoder(n_components=30)).
        Features classified under this category are imputed based on the
        strategy defined with `impute_missing`.

    numerical_transformer : {'drop', 'remainder', 'passthrough'} or Transformer, optional
        Transformer used on numerical features.
        Can either be a transformer object instance (e.g. StandardScaler),
        a Pipeline containing the preprocessing steps,
        'drop' for dropping the columns,
        'remainder' for applying `remainder`,
        'passthrough' to return the unencoded columns,
        or `None` to use the default transformer (here nothing, so 'passthrough').
        Features classified under this category are not imputed at all
        (regardless of `impute_missing`).

    datetime_transformer : {'drop', 'remainder', 'passthrough'} or Transformer, optional
        Transformer used on datetime features.
        Can either be a transformer object instance (e.g. DatetimeEncoder),
        a Pipeline containing the preprocessing steps,
        'drop' for dropping the columns,
        'remainder' for applying `remainder`,
        'passthrough' to return the unencoded columns,
        or `None` to use the default transformer (DatetimeEncoder()).
        Features classified under this category are not imputed at all
        (regardless of `impute_missing`).

    auto_cast : bool, optional, default=True
        If set to `True`, will try to convert each column to the best possible
        data type (dtype).

    impute_missing : {'auto', 'force', 'skip'}, default='auto'
        When to impute missing values in categorical (textual) columns.
        'auto' will impute missing values if it is considered appropriate
        (we are using an encoder that does not support missing values and/or
        specific versions of pandas, numpy and scikit-learn).
        'force' will impute missing values in all categorical columns.
        'skip' will not impute at all.
        When imputed, missing values are replaced by the string 'missing'
        before being encoded.
        As imputation logic for numerical features can be quite intricate,
        it is left to the user to manage.
        See also attribute :attr:`~skrub.TableVectorizer.imputed_columns_`.

    remainder : {'drop', 'passthrough'} or Transformer, default='passthrough'
        By default, all remaining columns that were not specified in `transformers`
        will be automatically passed through. This subset of columns is concatenated
        with the output of the transformers. (default 'passthrough').
        By specifying `remainder='drop'`, only the specified columns
        in `transformers` are transformed and combined in the output, and the
        non-specified columns are dropped.
        By setting `remainder` to be an estimator, the remaining
        non-specified columns will use the `remainder` estimator. The
        estimator must support :term:`fit` and :term:`transform`.
        Note that using this feature requires that the DataFrame columns
        input at :term:`fit` and :term:`transform` have identical order.

    sparse_threshold : float, default=0.0
        If the output of the different transformers contains sparse matrices,
        these will be stacked as a sparse matrix if the overall density is
        lower than this value. Use `sparse_threshold=0` to always return dense.
        When the transformed output consists of all dense data, the stacked
        result will be dense, and this keyword will be ignored.

    n_jobs : int, optional
        Number of jobs to run in parallel.
        ``None`` (the default) means 1 unless in a
        joblib.parallel_backend context.
        ``-1`` means using all processors.

    transformer_weights : dict, optional
        Multiplicative weights for features per transformer. The output of the
        transformer is multiplied by these weights. Keys are transformer names,
        values the weights.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    Attributes
    ----------
    transformers_ : list of 3-tuples (str, Transformer or str, list of str)
        The collection of fitted transformers as tuples of
        (name, fitted_transformer, column). `fitted_transformer` can be an
        estimator, 'drop', or 'passthrough'. In case there were no columns
        selected, this will be an unfitted transformer.
        If there are remaining columns, the final element is a tuple of the
        form:
        ('remainder', transformer, remaining_columns) corresponding to the
        `remainder` parameter. If there are remaining columns, then
        ``len(transformers_)==len(transformers)+1``, otherwise
        ``len(transformers_)==len(transformers)``.

    columns_ : pandas.Index
        The fitted array's columns. They are applied to the data passed
        to the `transform` method.

    types_ : dict mapping of str to type
        A mapping of inferred types per column.
        Key is the column name, value is the inferred dtype.
        Exists only if `auto_cast=True`.

    imputed_columns_ : list of str
        The list of columns in which we imputed the missing values.

    See Also
    --------
    GapEncoder :
        Encodes dirty categories (strings) by constructing latent topics with continuous encoding.
    MinHashEncoder :
        Encode string columns as a numeric array with the minhash method.
    SimilarityEncoder :
        Encode string columns as a numeric array with n-gram string similarity.

    Notes
    -----
    The column order of the input data is not guaranteed to be the same
    as the output data (returned by TableVectorizer.transform).
    This is a due to the way the ColumnTransformer works.
    However, the output column order will always be the same for different
    calls to TableVectorizer.transform on a same fitted TableVectorizer instance.
    For example, if input data has columns ['name', 'job', 'year'], then output
    columns might be shuffled, e.g. ['job', 'year', 'name'], but every call
    to TableVectorizer.transform on this instance will return this order.

    Examples
    --------
    Fit a TableVectorizer on an example dataset:

    >>> from skrub.datasets import fetch_employee_salaries
    >>> ds = fetch_employee_salaries()
    >>> ds.X.head(3)
      gender department                          department_name                                           division assignment_category      employee_position_title underfilled_job_title date_first_hired  year_first_hired
    0      F        POL                     Department of Police  MSB Information Mgmt and Tech Division Records...    Fulltime-Regular  Office Services Coordinator                   NaN       09/22/1986              1986
    1      M        POL                     Department of Police         ISB Major Crimes Division Fugitive Section    Fulltime-Regular        Master Police Officer                   NaN       09/12/1988              1988
    2      F        HHS  Department of Health and Human Services      Adult Protective and Case Management Services    Fulltime-Regular             Social Worker IV                   NaN       11/19/1989              1989

    >>> tv = TableVectorizer()
    >>> tv.fit(ds.X)

    Now, we can inspect the transformers assigned to each column:

    >>> tv.transformers_
    [
        ('datetime', DatetimeEncoder(), ['date_first_hired']),
        ('low_card_cat', OneHotEncoder(drop='if_binary', handle_unknown='ignore'),
         ['gender', 'department', 'department_name', 'assignment_category']),
        ('high_card_cat', GapEncoder(n_components=30),
         ['division', 'employee_position_title', 'underfilled_job_title']),
        ('remainder', 'passthrough', ['year_first_hired'])
    ]
    """

    transformers_: list[tuple[str, str | TransformerMixin, list[str]]]
    columns_: pd.Index
    types_: dict[str, type]
    imputed_columns_: list[str]

    # Override required parameters
    _required_parameters = []

    def __init__(
        self,
        *,
        cardinality_threshold: int = 40,
        low_card_cat_transformer: OptionalTransformer = None,
        high_card_cat_transformer: OptionalTransformer = None,
        numerical_transformer: OptionalTransformer = None,
        datetime_transformer: OptionalTransformer = None,
        auto_cast: bool = True,
        impute_missing: Literal["auto", "force", "skip"] = "auto",
        # The next parameters are inherited from ColumnTransformer
        remainder: Literal["drop", "passthrough"] | TransformerMixin = "passthrough",
        sparse_threshold: float = 0.0,
        n_jobs: int = None,
        transformer_weights=None,
        verbose: bool = False,
    ):
        super().__init__(transformers=[])

        self.cardinality_threshold = cardinality_threshold
        self.low_card_cat_transformer = low_card_cat_transformer
        self.high_card_cat_transformer = high_card_cat_transformer
        self.numerical_transformer = numerical_transformer
        self.datetime_transformer = datetime_transformer
        self.auto_cast = auto_cast
        self.impute_missing = impute_missing

        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose

    def _more_tags(self):
        """
        Used internally by sklearn to ease the estimator checks.
        """
        return {
            "X_types": ["2darray", "string"],
            "allow_nan": [True],
            "_xfail_checks": {
                "check_complex_data": "Passthrough complex columns as-is.",
                "check_dont_overwrite_parameters": (
                    "`transformers` is modified during fit but it is not pass by the"
                    " user. We need to create an instance of ColumnTransformer instead"
                    " but the current behaviour is not leading to a bug."
                ),
            },
        }

    def _clone_transformers(self):
        """
        For each of the different transformers that can be passed,
        create the corresponding variable name with a trailing underscore,
        which is the value that will be used in `transformers`.
        We clone the instances to avoid altering them.
        See the clone function docstring.
        Note: typos are not detected here, they are left in and are detected
        down the line in ColumnTransformer.fit_transform.
        """
        if isinstance(self.low_card_cat_transformer, sklearn.base.TransformerMixin):
            self.low_card_cat_transformer_ = clone(self.low_card_cat_transformer)
        elif self.low_card_cat_transformer is None:
            # sklearn is lenient and lets us use both
            # `handle_unknown="infrequent_if_exist"` and `drop="if_binary"`
            # at the same time
            self.low_card_cat_transformer_ = OneHotEncoder(
                drop="if_binary", handle_unknown="infrequent_if_exist"
            )
        elif self.low_card_cat_transformer == "remainder":
            self.low_card_cat_transformer_ = (
                self.remainder
                if isinstance(self.remainder, str)
                else clone(self.remainder)
            )
        else:
            self.low_card_cat_transformer_ = self.low_card_cat_transformer

        if isinstance(self.high_card_cat_transformer, sklearn.base.TransformerMixin):
            self.high_card_cat_transformer_ = clone(self.high_card_cat_transformer)
        elif self.high_card_cat_transformer is None:
            self.high_card_cat_transformer_ = GapEncoder(n_components=30)
        elif self.high_card_cat_transformer == "remainder":
            self.high_card_cat_transformer_ = (
                self.remainder
                if isinstance(self.remainder, str)
                else clone(self.remainder)
            )
        else:
            self.high_card_cat_transformer_ = self.high_card_cat_transformer

        if isinstance(self.numerical_transformer, sklearn.base.TransformerMixin):
            self.numerical_transformer_ = clone(self.numerical_transformer)
        elif self.numerical_transformer is None:
            self.numerical_transformer_ = "passthrough"
        elif self.numerical_transformer == "remainder":
            self.numerical_transformer_ = (
                self.remainder
                if isinstance(self.remainder, str)
                else clone(self.remainder)
            )
        else:
            self.numerical_transformer_ = self.numerical_transformer

        if isinstance(self.datetime_transformer, sklearn.base.TransformerMixin):
            self.datetime_transformer_ = clone(self.datetime_transformer)
        elif self.datetime_transformer is None:
            self.datetime_transformer_ = DatetimeEncoder()
        elif self.datetime_transformer == "remainder":
            self.datetime_transformer_ = (
                self.remainder
                if isinstance(self.remainder, str)
                else clone(self.remainder)
            )
        else:
            self.datetime_transformer_ = self.datetime_transformer

        # TODO: check that the provided transformers are valid

    def _auto_cast(self, X: pd.DataFrame) -> pd.DataFrame:
        """Takes a dataframe and tries to convert its columns to their best possible data type.

        Parameters
        ----------
        X : :obj:`~pandas.DataFrame` of shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        :obj:`~pandas.DataFrame`
            The same :obj:`~pandas.DataFrame`, with its columns cast to their
            best possible data type.
        """
        # Handle missing values
        for col in X.columns:
            # Convert pandas' NaN value (pd.NA) to numpy NaN value (np.nan)
            # because the former tends to raise all kind of issues when dealing
            # with scikit-learn (as of version 0.24).
            if _has_missing_values(X[col]):
                # Some numerical dtypes like Int64 or Float64 only support
                # pd.NA, so they must be converted to np.float64 before.
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].astype(np.float64)
                X[col].fillna(value=np.nan, inplace=True)

        # for object dtype columns, first convert to string to avoid mixed types
        object_cols = X.columns[X.dtypes == "object"]
        for col in object_cols:
            X[col] = np.where(X[col].isna(), X[col], X[col].astype(str))

        # Convert to the best possible data type
        self.types_ = {}
        for col in X.columns:
            if not pd.api.types.is_datetime64_any_dtype(X[col]):
                # we don't want to cast datetime64
                try:
                    X[col] = pd.to_numeric(X[col], errors="raise")
                except (ValueError, TypeError):
                    # Only try to convert to datetime
                    # if the variable isn't numeric.
                    # try to find the best format
                    format = _infer_date_format(X[col])
                    # if a format is found, try to apply to the whole column
                    # if no format is found, pandas will try to parse each row
                    # with a different engine, which can understand weirder formats
                    try:
                        # catch the warnings raised by pandas
                        # in case the conversion fails
                        with warnings.catch_warnings(record=True) as w:
                            X[col] = pd.to_datetime(
                                X[col], errors="raise", format=format
                            )
                        # if the conversion worked, raise pandas warnings
                        for warning in w:
                            with warnings.catch_warnings():
                                # otherwise the warning is considered a duplicate
                                warnings.simplefilter("always")
                                warnings.warn(
                                    "Warning raised by pandas when converting column"
                                    f" '{col}' to datetime: "
                                    + str(warning.message),
                                    UserWarning,
                                    stacklevel=2,
                                )
                    except (ValueError, TypeError):
                        pass
            # Cast pandas dtypes to numpy dtypes
            # for earlier versions of sklearn. FIXME: which ?
            if issubclass(X[col].dtype.__class__, ExtensionDtype):
                try:
                    X[col] = X[col].astype(X[col].dtype.type, errors="ignore")
                except (TypeError, ValueError):
                    pass
            self.types_[col] = X[col].dtype
        return X

    def _apply_cast(self, X: pd.DataFrame) -> pd.DataFrame:
        """Takes a dataframe, and applies the best data types learnt during fitting.

        Does the same thing as `_auto_cast`, but applies learnt info.
        """
        for col in X.columns:
            X[col] = _replace_false_missing(X[col])
            if _has_missing_values(X[col]):
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].astype(np.float64)
                X[col].fillna(value=np.nan, inplace=True)
        for col in self.imputed_columns_:
            X[col] = _replace_missing_in_cat_col(X[col])
        # for object dtype columns, first convert to string to avoid mixed types
        # we do it both in auto_cast and apply_cast because
        # the type infered for string columns during auto_cast
        # is not necessarily string, it can be object because
        # of missing values
        object_cols = X.columns[X.dtypes == "object"]
        for col in object_cols:
            X[col] = np.where(X[col].isna(), X[col], X[col].astype(str))
        for col, dtype in self.types_.items():
            # if categorical, add the new categories to prevent
            # them to be encoded as nan
            if isinstance(dtype, pd.CategoricalDtype):
                known_categories = dtype.categories
                new_categories = pd.unique(X[col])
                # remove nan from new_categories
                new_categories = new_categories[~pd.isnull(new_categories)]
                dtype = pd.CategoricalDtype(
                    categories=known_categories.union(new_categories)
                )
                self.types_[col] = dtype
        for col, dtype in self.types_.items():
            try:
                if pd.api.types.is_numeric_dtype(dtype):
                    # we don't use astype because it can convert float to int
                    X[col] = pd.to_numeric(X[col])
                else:
                    X[col] = X[col].astype(dtype)
            except ValueError as e:
                culprit = parse_astype_error_message(e)
                if culprit is None:
                    raise e
                warnings.warn(
                    f"Value '{culprit}' could not be converted to infered type"
                    f" {dtype!s} in column '{col}'. Such values will be replaced"
                    " by NaN.",
                    UserWarning,
                    stacklevel=2,
                )
                # if the inferred dtype is numerical or datetime,
                # we want to ignore entries that cannot be converted
                # to this dtype
                if pd.api.types.is_numeric_dtype(dtype):
                    X[col] = pd.to_numeric(X[col], errors="coerce")
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    X[col] = pd.to_datetime(X[col], errors="coerce")
                else:
                    # this should not happen
                    raise e
        return X

    def _check_X(self, X, reset):
        if sparse.isspmatrix(X):
            raise TypeError(
                "A sparse matrix was passed, but dense data is required. Use "
                "X.toarray() to convert to a dense numpy array."
            )

        if not isinstance(X, pd.DataFrame):
            # check the dimension of X before to create a dataframe that always
            # `ndim == 2`
            # unfortunately, we need to call `asarray` before to call `ndim`
            # in case that the container implement `__array_function__`
            X_array = np.asarray(X)
            if X_array.ndim == 0:
                raise ValueError(
                    f"Expected 2D array, got scalar array instead:\narray={X}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample."
                )
            if X_array.ndim == 1:
                raise ValueError(
                    f"Expected 2D array, got 1D array instead:\narray={X}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample."
                )
            X = pd.DataFrame(X_array)
        else:
            # Create a copy to avoid altering the original data.
            X = X.copy()

        if X.shape[0] < 1:
            raise ValueError(
                f"Found array with {X.shape[0]} sample(s) (shape={X.shape}) while a"
                " minimum of 1 is required."
            )
        if X.shape[1] < 1:
            raise ValueError(
                f"Found array with {X.shape[1]} feature(s) (shape={X.shape}) while"
                " a minimum of 1 is required."
            )
        self._check_n_features(X, reset=reset)

        return X

    def fit_transform(self, X: ArrayLike, y: ArrayLike = None) -> ArrayLike:
        """Fit all transformers, transform the data, and concatenate the results.

        In practice, it (1) converts features to their best possible types
        if `auto_cast=True`, (2) classify columns based on their data type,
        (3) replaces "false missing" (see _replace_false_missing),
        and imputes categorical columns depending on `impute_missing`, and
        finally, transforms `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data, of which specified subsets are used to fit the
            transformers.
        y : array-like of shape (n_samples,), optional
            Targets for supervised learning.

        Returns
        -------
        {array-like, sparse matrix} of shape (n_samples, sum_n_components)
            Hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers. If
            any result is a sparse matrix, everything will be converted to
            sparse matrices.
        """
        if self.impute_missing not in ("skip", "force", "auto"):
            raise ValueError(
                "Invalid value for `impute_missing`, expected any of "
                "{'auto', 'force', 'skip'}, "
                f"got {self.impute_missing!r}. "
            )

        self._clone_transformers()

        X = self._check_X(X, reset=True)

        self.columns_ = X.columns

        # We replace in all columns regardless of their type,
        # as we might have some false missing
        # in numerical columns for instance.
        X = _replace_false_missing(X)

        ###
        # We need to check for duplicate column names.
        # It is checked by comparing the number of unique values
        # to the length of the column names
        if len(set(X.columns)) != len(X.columns):
            raise AssertionError(
                "Duplicate column names in the dataframe."
                f"The column names are {X.columns}"
            )
        ###

        # If auto_cast is True, we'll find and apply the best possible type
        # to each column.
        # We'll keep the results in order to apply the types in `transform`.
        if self.auto_cast:
            X = self._auto_cast(X)

        # Select columns by dtype
        numeric_columns = X.select_dtypes(include="number").columns.to_list()
        categorical_columns = X.select_dtypes(
            include=["string", "object", "category"]
        ).columns.to_list()
        datetime_columns = X.select_dtypes(
            include=["datetime", "datetimetz"]
        ).columns.to_list()

        # Classify categorical columns by cardinality
        low_card_cat_columns, high_card_cat_columns = [], []
        for col in categorical_columns:
            if X[col].nunique() < self.cardinality_threshold:
                low_card_cat_columns.append(col)
            else:
                high_card_cat_columns.append(col)

        # Next part: construct the transformers
        # Create the list of all the transformers.
        all_transformers: list[tuple[str, OptionalTransformer, list[str]]] = [
            ("numeric", self.numerical_transformer_, numeric_columns),
            ("datetime", self.datetime_transformer_, datetime_columns),
            ("low_card_cat", self.low_card_cat_transformer_, low_card_cat_columns),
            ("high_card_cat", self.high_card_cat_transformer_, high_card_cat_columns),
        ]
        # We will now filter this list, by keeping only the ones with:
        # - at least one column
        # - a valid encoder or string (filter out if None)
        self.transformers = []
        for trans in all_transformers:
            name, enc, cols = trans  # Unpack
            if len(cols) > 0 and enc is not None:
                self.transformers.append(trans)

        self.imputed_columns_ = []
        if self.impute_missing != "skip":
            # Impute if suiting
            if _has_missing_values(X):
                if self.impute_missing == "force":
                    # Only impute categorical columns
                    for col in categorical_columns:
                        X[col] = _replace_missing_in_cat_col(X[col])
                        self.imputed_columns_.append(col)

                elif self.impute_missing == "auto":
                    # Add special cases when we should impute.
                    pass

        # If there was missing values imputation, we cast the DataFrame again,
        # as pandas gives different types depending on whether a column has
        # missing values or not.
        if self.imputed_columns_ and self.auto_cast:
            X = self._auto_cast(X)

        if self.verbose:
            print(f"[TableVectorizer] Assigned transformers: {self.transformers}")

        X_enc = super().fit_transform(X, y)

        # For the "remainder" columns, the `ColumnTransformer` `transformers_`
        # attribute contains the index instead of the column name,
        # so we convert the values to the appropriate column names
        # if there is less than 20 columns in the remainder.
        for i, (name, enc, cols) in enumerate(self.transformers_):
            if name == "remainder" and len(cols) < 20:
                # In this case, "cols" is a list of ints (the indices)
                cols: list[int]
                self.transformers_[i] = (name, enc, [self.columns_[j] for j in cols])

        return X_enc

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Transform `X` by applying the fitted transformers on the columns.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        {array-like, sparse matrix} of shape (n_samples, sum_n_components)
            Hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers. If
            any result is a sparse matrix, everything will be converted to
            sparse matrices.
        """
        check_is_fitted(self, attributes=["transformers_"])

        X = self._check_X(X, reset=False)

        if (X.columns != self.columns_).all():
            X.columns = self.columns_

        if self.auto_cast:
            X = self._apply_cast(X)

        return super().transform(X)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Return clean feature names.

        Feature names are formatted like:
        "<column_name>_<value>" if encoded by OneHotEncoder or alike,
        (e.g. "job_title_Police officer"), or "<column_name>" otherwise.

        Parameters
        ----------
        input_features : None
            Unused, only here for compatibility.

        Returns
        -------
        list of str
            Feature names.
        """
        ct_feature_names = super().get_feature_names_out()
        all_trans_feature_names = []

        for name, trans, cols, _ in self._iter(fitted=True):
            if isinstance(trans, str):
                if trans == "drop":
                    continue
                elif trans == "passthrough":
                    if all(isinstance(col, int) for col in cols):
                        cols = [self.columns_[i] for i in cols]
                    all_trans_feature_names.extend(cols)
                continue
            trans_feature_names = trans.get_feature_names_out(cols)
            all_trans_feature_names.extend(trans_feature_names)

        if len(ct_feature_names) != len(all_trans_feature_names):
            warn("Could not extract clean feature names; returning defaults. ")
            return list(ct_feature_names)

        return all_trans_feature_names


@deprecated("Use TableVectorizer instead.")
class SuperVectorizer(TableVectorizer):
    """Deprecated name of TableVectorizer."""

    pass
