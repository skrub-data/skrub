"""
Implements the TableVectorizer: a preprocessor to automatically apply
transformers/encoders to different types of data, without the need to
manually categorize them beforehand, or construct complex Pipelines.
"""

import warnings
from collections import Counter

import numpy as np
import pandas as pd
import sklearn
from pandas._libs.tslibs.parsing import guess_datetime_format
from pandas.core.dtypes.base import ExtensionDtype
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.compose._column_transformer import _get_transformer_list
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from skrub import DatetimeEncoder, GapEncoder
from skrub._utils import clone_if_default, parse_astype_error_message

HIGH_CARDINALITY_TRANSFORMER = GapEncoder(n_components=30)
LOW_CARDINALITY_TRANSFORMER = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore",
    drop="if_binary",
)
DATETIME_TRANSFORMER = DatetimeEncoder()


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
                    Both {date_format_monthfirst.iloc[0]} and
                    {date_format_dayfirst.iloc[0]} are valid formats for the dates in
                    column '{date_column.name}'.
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


def _clone_during_fit(transformer, remainder, n_jobs):
    if isinstance(transformer, sklearn.base.TransformerMixin):
        return _propagate_n_jobs(clone(transformer), n_jobs)
    elif transformer == "remainder":
        return remainder if isinstance(remainder, str) else clone(remainder)
    elif transformer == "passthrough":
        return transformer
    else:
        raise ValueError(
            "'transformer' must be an instance of sklearn.base.TransformerMixin, "
            f"'remainder' or 'passthrough'. Got {transformer=!r}."
        )


def _check_specific_transformers(specific_transformers, n_jobs):
    if (specific_transformers is None) or len(specific_transformers) == 0:
        return []
    else:
        first_item_length = len(specific_transformers[0])
        # Check that all tuples have the same length
        for idx, tuple_ in enumerate(specific_transformers):
            if len(tuple_) != first_item_length:
                raise TypeError(
                    "Expected `specific_transformers` to be a list of "
                    "tuples with all the same length, got length "
                    f"{len(tuple_)} at index {idx} (elements at previous "
                    f"indices have {first_item_length} in length). "
                )
        if first_item_length == 2:
            # Unnamed assignments, transform to named
            specific_transformers = _get_transformer_list(specific_transformers)
        elif first_item_length == 3:
            # Named assignments, no-op
            pass
        else:
            raise TypeError(
                "Expected `specific_transformers` to be a list of tuples "
                "of length 2 or 3, got a list of tuples of length "
                f"{first_item_length}. "
            )

        return [
            (
                (name, _propagate_n_jobs(clone(transformer), n_jobs), cols)
                if isinstance(transformer, sklearn.base.TransformerMixin)
                else (name, transformer, cols)
            )
            for name, transformer, cols in specific_transformers
        ]


def _propagate_n_jobs(transformer, n_jobs):
    if n_jobs is not None and (
        hasattr(transformer, "n_jobs") and transformer.n_jobs is None
    ):
        transformer.set_params(n_jobs=n_jobs)
    return transformer


class TableVectorizer(TransformerMixin, BaseEstimator):
    """Automatically transform a heterogeneous dataframe to a numerical array.

    Easily transforms a heterogeneous data table
    (such as a :obj:`pandas.DataFrame`) to a numerical array for machine
    learning. To do so, the TableVectorizer transforms each column depending
    on its data type.

    Parameters
    ----------
    cardinality_threshold : int, default=40
        Two lists of features will be created depending on this value: strictly
        under this value, the low cardinality categorical features, and above or
        equal, the high cardinality categorical features.
        Different transformers will be applied to these two groups,
        defined by the parameters `low_cardinality_transformer` and
        `high_cardinality_transformer` respectively.
        Note: currently, missing values are counted as a single unique value
        (so they count in the cardinality).

    low_cardinality_transformer : {'drop', 'remainder', 'passthrough'} \
        or Transformer, optional
        Transformer used on categorical/string features with low cardinality
        (threshold is defined by `cardinality_threshold`).
        Can either be a transformer object instance (e.g. OneHotEncoder),
        a Pipeline containing the preprocessing steps,
        'drop' for dropping the columns,
        'remainder' for applying `remainder`,
        'passthrough' to return the unencoded columns.
        The default transformer is \
            ``OneHotEncoder(handle_unknown="ignore", drop="if_binary")``.
        Features classified under this category are imputed based on the
        strategy defined with `impute_missing`.

    high_cardinality_transformer : {'drop', 'remainder', 'passthrough'} \
        or Transformer, optional
        Transformer used on categorical/string features with high cardinality
        (threshold is defined by `cardinality_threshold`).
        Can either be a transformer object instance
        (e.g. GapEncoder), a Pipeline containing the preprocessing steps,
        'drop' for dropping the columns,
        'remainder' for applying `remainder`,
        or 'passthrough' to return the unencoded columns.
        The default transformer is ``GapEncoder(n_components=30)``.
        Features classified under this category are imputed based on the
        strategy defined with `impute_missing`.

    numerical_transformer : {'drop', 'remainder', 'passthrough'} \
        or Transformer, optional
        Transformer used on numerical features.
        Can either be a transformer object instance (e.g. StandardScaler),
        a Pipeline containing the preprocessing steps,
        'drop' for dropping the columns,
        'remainder' for applying `remainder`,
        or 'passthrough' to return the unencoded columns (default).
        Features classified under this category are not imputed at all
        (regardless of `impute_missing`).

    datetime_transformer : {'drop', 'remainder', 'passthrough'} or Transformer, optional
        Transformer used on datetime features.
        Can either be a transformer object instance (e.g. DatetimeEncoder),
        a Pipeline containing the preprocessing steps,
        'drop' for dropping the columns,
        'remainder' for applying `remainder`,
        'passthrough' to return the unencoded columns.
        The default transformer is ``DatetimeEncoder()``.
        Features classified under this category are not imputed at all
        (regardless of `impute_missing`).

    specific_transformers : list of tuples ({'drop', 'remainder', 'passthrough'} or \
        Transformer, list of str or int) or (str, {'drop', 'remainder', 'passthrough'} \
            or Transformer, list of str or int), optional
        On top of the default column type classification (see parameters above),
        this parameter allows you to manually specify transformers for
        specific columns.
        This is equivalent to using a ColumnTransformer for assigning the
        column-specific transformers, and passing the ``TableVectorizer``
        as the ``remainder``.
        This parameter can take two different formats, either:
        - a list of 2-tuples (transformer, column names or indices)
        - a list of 3-tuple (name, transformer, column names or indices)
        In the latter format, you can specify the name of the assignment.
        Mixing the two is not supported.

    auto_cast : bool, default=True
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

    n_jobs : int, default=None
        Number of jobs to run in parallel. This number of jobs will be dispatched to
        the underlying transformers, if those support parallelization and they do not
        set specifically ``n_jobs``.
        ``None`` (the default) means 1 unless in a :func:`joblib.parallel_config`
        context. ``-1`` means using all processors.

    transformer_weights : dict, default=None
        Multiplicative weights for features per transformer. The output of the
        transformer is multiplied by these weights. Keys are transformer names,
        values the weights.

    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed.

    verbose_feature_names_out : bool, default=False
        If True, :meth:`TableVectorizer.get_feature_names_out` will prefix
        all feature names with the name of the transformer that generated that
        feature.
        If False, :meth:`TableVectorizer.get_feature_names_out` will not
        prefix any feature names and will error if feature names are not
        unique.

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

    types_ : dict mapping of int to type
        A mapping of inferred types per column.
        Key is the index of a column, value is the inferred dtype.
        Exists only if `auto_cast=True`.

    imputed_columns_ : list of str
        The list of columns in which we imputed the missing values.

    See Also
    --------
    GapEncoder :
        Encodes dirty categories (strings) by constructing latent topics with \
            continuous encoding.
    MinHashEncoder :
        Encode string columns as a numeric array with the minhash method.
    SimilarityEncoder :
        Encode string columns as a numeric array with n-gram string similarity.

    Notes
    -----
    The column order of the input data is not guaranteed to be the same
    as the output data (returned by TableVectorizer.transform).
    This is a due to the way the underlying ColumnTransformer works.
    However, the output column order will always be the same for different
    calls to ``TableVectorize.transform`` on a same fitted TableVectorizer instance.
    For example, if input data has columns ['name', 'job', 'year'], then output
    columns might be shuffled, e.g. ['job', 'year', 'name'], but every call
    to ``TableVectorizer.transform`` on this instance will return this order.

    Examples
    --------
    Fit a TableVectorizer on an example dataset:

    >>> from skrub.datasets import fetch_employee_salaries
    >>> ds = fetch_employee_salaries()
    >>> ds.X.head(3)
      gender department  ... date_first_hired year_first_hired
    0      F        POL  ...       09/22/1986             1986
    1      M        POL  ...       09/12/1988             1988
    2      F        HHS  ...       11/19/1989             1989
    [3 rows x 8 columns]

    >>> tv = TableVectorizer()
    >>> tv.fit(ds.X)
    TableVectorizer()

    Now, we can inspect the transformers assigned to each column:

    >>> tv.transformers_
    [('numeric', 'passthrough', ['year_first_hired']), \
('datetime', DatetimeEncoder(), ['date_first_hired']), \
('low_card_cat', OneHotEncoder(drop='if_binary', handle_unknown='ignore', \
sparse_output=False), \
['gender', 'department', 'department_name', 'assignment_category']), \
('high_card_cat', GapEncoder(n_components=30), ['division', 'employee_position_title'])]
    """

    def __init__(
        self,
        *,
        cardinality_threshold=40,
        low_cardinality_transformer=LOW_CARDINALITY_TRANSFORMER,
        high_cardinality_transformer=HIGH_CARDINALITY_TRANSFORMER,
        numerical_transformer="passthrough",
        datetime_transformer=DATETIME_TRANSFORMER,
        specific_transformers=None,
        auto_cast=True,
        impute_missing="auto",
        # The next parameters are inherited from ColumnTransformer
        remainder="passthrough",
        sparse_threshold=0.0,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=False,
    ):
        self.cardinality_threshold = cardinality_threshold
        self.low_cardinality_transformer = clone_if_default(
            low_cardinality_transformer, LOW_CARDINALITY_TRANSFORMER
        )
        self.high_cardinality_transformer = clone_if_default(
            high_cardinality_transformer, HIGH_CARDINALITY_TRANSFORMER
        )
        self.datetime_transformer = clone_if_default(
            datetime_transformer, DATETIME_TRANSFORMER
        )
        self.numerical_transformer = numerical_transformer
        self.specific_transformers = specific_transformers
        self.auto_cast = auto_cast
        self.impute_missing = impute_missing

        # Parameter from `ColumnTransformer`
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose
        self.verbose_feature_names_out = verbose_feature_names_out

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
        for transformer_name in [
            "high_cardinality_transformer",
            "low_cardinality_transformer",
            "datetime_transformer",
            "numerical_transformer",
        ]:
            transformer = _clone_during_fit(
                getattr(self, transformer_name),
                remainder=self.remainder,
                n_jobs=self.n_jobs,
            )
            setattr(self, f"{transformer_name}_", transformer)

        self.specific_transformers_ = _check_specific_transformers(
            self.specific_transformers,
            self.n_jobs,
        )

    def _auto_cast(self, X: pd.DataFrame) -> pd.DataFrame:
        """Takes a dataframe and tries to convert its columns to their best possible
        data type.

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
        for col_idx, col in enumerate(X.columns):
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
            self.types_[col_idx] = X[col].dtype
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
        # for object dtype columns, first convert to string to avoid mixed
        # types we do it both in auto_cast and apply_cast because
        # the type inferred for string columns during auto_cast
        # is not necessarily string, it can be an object because
        # of missing values
        object_cols = X.columns[X.dtypes == "object"]
        for col in object_cols:
            X[col] = np.where(X[col].isna(), X[col], X[col].astype(str))
        for col_idx, dtype in self.types_.items():
            col = X.columns[col_idx]
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
                self.types_[col_idx] = dtype
        for col_idx, dtype in self.types_.items():
            col = X.columns[col_idx]
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
                    f"Value '{culprit}' could not be converted to inferred type"
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

    def _check_X(self, X):
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
            feature_names = getattr(self, "feature_names_in_", None)
            X = pd.DataFrame(X_array, columns=feature_names)
        else:
            # Create a copy to avoid altering the original data.
            X = X.copy()

        # Check Pandas sparse arrays
        is_sparse_col = [hasattr(X[col], "sparse") for col in X.columns]
        if any(is_sparse_col):
            sparse_cols = X.columns[is_sparse_col]
            raise TypeError(
                f"Columns {sparse_cols!r} are sparse Pandas series, but dense "
                "data is required. Use df[col].sparse.to_dense() to convert "
                "a series from sparse to dense."
            )

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
        return X

    def fit(self, X, y=None):
        """Fit all transformers using X.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            Input data, of which specified subsets are used to fit the
            transformers.

        y : array-like of shape (n_samples, ...), default=None
            Targets for supervised learning.

        Returns
        -------
        self : TableVectorizer
            This estimator.
        """
        # we use fit_transform to make sure to set sparse_output_ (for which we
        # need the transformed data) to have consistent output type in predict
        self.fit_transform(X, y=y)
        return self

    def fit_transform(self, X, y=None):
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

        self._check_feature_names(X, reset=True)
        X = self._check_X(X)
        self._check_n_features(X, reset=True)

        # We replace in all columns regardless of their type,
        # as we might have some false missing
        # in numerical columns for instance.
        X = _replace_false_missing(X)

        # Check for duplicate column names.
        duplicate_columns = {k for k, v in Counter(X.columns).items() if v > 1}
        if duplicate_columns:
            raise AssertionError(
                f"Duplicate column names in the dataframe: {duplicate_columns}"
            )

        # If auto_cast is True, we'll find and apply the best possible type
        # to each column.
        # We'll keep the results in order to apply the types in `transform`.
        if self.auto_cast:
            X = self._auto_cast(X)

        # We will filter X to keep only the columns that are not specified
        # explicitly by the user.
        X_filtered = X.drop(
            columns=[
                # We do this for loop as `self.specific_transformers_`
                # might be empty.
                col
                for (_, _, columns) in self.specific_transformers_
                for col in columns
            ]
        )
        # Select columns by dtype
        numeric_columns = X_filtered.select_dtypes(include="number").columns.to_list()
        categorical_columns = X_filtered.select_dtypes(
            include=["string", "object", "category"]
        ).columns.to_list()
        datetime_columns = X_filtered.select_dtypes(
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
        all_transformers = [
            ("numeric", self.numerical_transformer_, numeric_columns),
            ("datetime", self.datetime_transformer_, datetime_columns),
            ("low_card_cat", self.low_cardinality_transformer_, low_card_cat_columns),
            (
                "high_card_cat",
                self.high_cardinality_transformer_,
                high_card_cat_columns,
            ),
            *self.specific_transformers_,
        ]
        # We will now filter this list, by keeping only the ones with:
        # - at least one column
        # - a valid encoder or string (filter out if None)
        transformers = []
        for name, transformer, columns in all_transformers:
            if len(columns) > 0 and transformer is not None:
                transformers.append((name, transformer, columns))

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
            print(f"[TableVectorizer] Assigned transformers: {transformers}")

        self._column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder=self.remainder,
            sparse_threshold=self.sparse_threshold,
            n_jobs=1,  # we don't parallelize the outer loop
            transformer_weights=self.transformer_weights,
            verbose=self.verbose,
            verbose_feature_names_out=self.verbose_feature_names_out,
        )

        X_enc = self._column_transformer.fit_transform(X, y=y)

        return X_enc

    def transform(self, X):
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
        check_is_fitted(self, attributes=["_column_transformer"])

        X = self._check_X(X)

        if self.auto_cast:
            X = self._apply_cast(X)

        return self._column_transformer.transform(X)

    def get_feature_names_out(self, input_features=None):
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
        feature_names : ndarray of str
            Feature names.
        """
        return self._column_transformer.get_feature_names_out(input_features)

    @property
    def transformers_(self):
        """Transformers applied to the different columns."""
        # For the "remainder" columns, the `ColumnTransformer` `transformers_`
        # attribute contains the index instead of the column name,
        # so we convert the values to the appropriate column names
        # if there is less than 20 columns in the remainder.
        transformers = []
        for name, transformer, columns in self._column_transformer.transformers_:
            # TODO: potentially remove when
            # https://github.com/scikit-learn/scikit-learn/issues/27533 is resolved.
            if name == "remainder" and len(columns) < 20:
                columns = self.feature_names_in_[columns].tolist()
            transformers.append((name, transformer, columns))
        return transformers

    @property
    def named_transformers_(self) -> Bunch:
        """Map transformer names to transformer objects.

        Read-only attribute to access any transformer by given name.
        Keys are transformer names and values are the fitted transformer
        objects.
        """
        return self._column_transformer.named_transformers_

    @property
    def sparse_output_(self) -> bool:
        """Whether the output of ``transform`` is sparse or dense.

        Boolean flag indicating whether the output of ``transform`` is a
        sparse matrix or a dense numpy array, which depends on the output
        of the individual transformers and the `sparse_threshold` keyword.
        """
        return self._column_transformer.sparse_output_

    @property
    def output_indices_(self) -> dict[str, slice]:
        """Map the transformer names to their input indices.

        A dictionary from each transformer name to a slice, where the slice
        corresponds to indices in the transformed output. This is useful to
        inspect which transformer is responsible for which transformed
        feature(s).
        """
        return self._column_transformer.output_indices_

    def _more_tags(self) -> dict:
        """
        Used internally by sklearn to ease the estimator checks.
        """
        return {
            "X_types": ["2darray", "string"],
            "allow_nan": [True],
            "_xfail_checks": {
                "check_complex_data": "Passthrough complex columns as-is.",
            },
        }
