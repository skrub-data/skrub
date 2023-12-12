"""
Implements the TableVectorizer: a preprocessor to automatically apply
transformers/encoders to different types of data, without the need to
manually categorize them beforehand, or construct complex Pipelines.
"""

from collections import Counter

import numpy as np
import pandas as pd
from pandas.api.types import (
    CategoricalDtype,
    is_datetime64_any_dtype,
    is_extension_array_dtype,
    is_numeric_dtype,
    is_object_dtype,
)
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.compose._column_transformer import _get_transformer_list
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted

from skrub import DatetimeEncoder, GapEncoder, to_datetime
from skrub._utils import clone_if_default

HIGH_CARDINALITY_TRANSFORMER = GapEncoder(n_components=30)
LOW_CARDINALITY_TRANSFORMER = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore",
    drop="if_binary",
)
DATETIME_TRANSFORMER = DatetimeEncoder()


def _to_numeric(X):
    """Convert the columns of a dataframe into a numeric representation.

    Parameters
    ----------
    X : pandas.DataFrame

    Returns
    -------
    X : pandas.DataFrame
    """
    X_out = dict()
    for col in X.columns:
        if not is_datetime64_any_dtype(X[col]):
            # We don't use errors="ignore" because it casts string
            # and categories to object dtype in Pandas < 2.0.
            # TODO: replace 'raise' by 'ignore' and remove the exception
            # catching when the minimum pandas version of skrub is 2.0.
            try:
                X_out[col] = pd.to_numeric(X[col], errors="raise")
                continue
            except (ValueError, TypeError):
                pass
        X_out[col] = X[col]
    return pd.DataFrame(X_out, index=X.index)


def _replace_missing_indicators(column):
    """Replace missing indicators, e.g., #NA, with np.nan and returns a copy.

    Parameters
    ----------
    column : pandas.Series

    Returns
    -------
    column : pandas.Series
    """
    # Taken from pandas.io.parsers (version 1.1.4)
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
        None,
        "?",
        "...",
    ]
    # Also replaces the whitespaces
    column = column.replace(STR_NA_VALUES, np.nan).replace(r"^\s+$", np.nan, regex=True)
    return column


def _union_category(X_col, dtype):
    """Update a categorical dtype with new entries."""
    known_categories = dtype.categories
    new_categories = pd.unique(X_col.loc[X_col.notnull()])
    dtype = pd.CategoricalDtype(categories=known_categories.union(new_categories))
    return dtype


def _clone_during_fit(transformer, remainder, n_jobs):
    if isinstance(transformer, TransformerMixin):
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
                if isinstance(transformer, TransformerMixin)
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
        Can either be a:
        - transformer object instance (e.g. OneHotEncoder)
        - a Pipeline containing the preprocessing steps
        - 'drop' for dropping the columns
        - 'remainder' for applying `remainder`
        - 'passthrough' to return the unencoded columns

        The default transformer is
            ```
            OneHotEncoder(
                handle_unknown='ignore',
                drop='if_binary',
                sparse_output=False,
            )
            ```

        When the downstream estimator is a tree-based model
        (e.g., scikit-learn HistGradientBoostingRegressor), the OneHotEncoder
        may lead to lower performances than other transformers,
        such as the OrdinalEncoder.

    high_cardinality_transformer : {'drop', 'remainder', 'passthrough'} \
        or Transformer, optional
        Transformer used on categorical/string features with high cardinality
        (threshold is defined by `cardinality_threshold`).
        Can either be a transformer object instance
        (e.g. GapEncoder), a Pipeline containing the preprocessing steps,
        'drop' for dropping the columns,
        'remainder' for applying `remainder`,
        or 'passthrough' to return the unencoded columns.
        The default transformer is ``GapEncoder(n_components=30)``

    numerical_transformer : {'drop', 'remainder', 'passthrough'} \
        or Transformer, optional
        Transformer used on numerical features.
        Can either be a transformer object instance (e.g. StandardScaler),
        a Pipeline containing the preprocessing steps,
        'drop' for dropping the columns,
        'remainder' for applying `remainder`,
        or 'passthrough' to return the unencoded columns (default).

    datetime_transformer : {'drop', 'remainder', 'passthrough'} or Transformer, optional
        Transformer used on datetime features.
        Can either be a transformer object instance (e.g. DatetimeEncoder),
        a Pipeline containing the preprocessing steps,
        'drop' for dropping the columns,
        'remainder' for applying `remainder`,
        'passthrough' to return the unencoded columns.
        The default transformer is ``DatetimeEncoder()``.

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
        If set to ``True``, calling ``fit``, ``transform`` or ``fit_transform``
        will call ``_auto_cast`` to convert each column to the "optimal" dtype
        for scikit-learn estimators.
        The main heuristics are the following:
        - pandas extension dtypes conversion to numpy dtype
        - datetime conversion using ``skrub.to_datetime``
        - numeric conversion using ``pandas.to_numeric``
        - numeric columns with missing values are converted to float to input np.nan
        - categorical columns dtypes are updated with the new entries (if any)
          during transform.

    remainder : {'drop', 'passthrough'} or Transformer, default='passthrough'
        By default, all remaining columns that were not specified in `transformers`
        will be automatically passed through. This subset of columns is concatenated
        with the output of the transformers. (default 'passthrough').
        By specifying ``remainder='drop'``, only the specified columns
        in `transformers` are transformed and combined in the output, and the
        non-specified columns are dropped.
        By setting `remainder` to be an estimator, the remaining
        non-specified columns will use the `remainder` estimator. The
        estimator must support ``fit`` and ``transform``.
        Note that using this feature requires that the DataFrame columns
        input at ``fit`` and ``transform`` have identical order.

    sparse_threshold : float, default=0.0
        If the output of the different transformers contains sparse matrices,
        these will be stacked as a sparse matrix if the overall density is
        lower than this value. Use ``sparse_threshold=0`` to always return dense.
        When the transformed output consists of all dense data, the stacked
        result will be dense, and this keyword will be ignored.
        Note that with the default encoders, the output will never be sparse.

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

    inferred_column_types_ : dict mapping of int to type
        A mapping of inferred types per column.

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
('low_cardinality', OneHotEncoder(drop='if_binary', handle_unknown='ignore', \
sparse_output=False), \
['gender', 'department', 'department_name', 'assignment_category']), \
('high_cardinality', GapEncoder(n_components=30), \
    ['division', 'employee_position_title'])]
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

    def _auto_cast(self, X, reset=True):
        """Convert each column of a dataframe to the "optimal" dtype \
            for scikit-learn estimators.

        The main heuristics are the following:
        - pandas extension dtypes conversion to numpy dtype
        - datetime conversion using ``skrub.to_datetime``
        - numeric conversion using ``pandas.to_numeric``
        - numeric columns with missing values are converted to float to input np.nan
        - categorical columns dtypes are updated with the new entries (if any)
          during transform.

        Parameters
        ----------
        X : :obj:`~pandas.DataFrame` of shape (n_samples, n_features)
            The data to be transformed.

        reset : bool, default=True
            If set to ``True`` (during fit), creates ``inferred_column_types_``,
            the mapping between columns of the training dataframe and their types.
            If set to ``False`` (during transform), updates ``inferred_column_types_``
            for the categorical columns with the new categories seen during transform.

        Returns
        -------
        X : :obj:`~pandas.DataFrame`
            The same :obj:`~pandas.DataFrame`, with its columns cast.
        """
        if self.auto_cast:
            for col in X.columns:
                X[col] = _replace_missing_indicators(X[col])

                # Some numerical dtypes like Int64 or Float64 only support
                # pd.NA, so they must be converted to np.float64 before imputing
                # with np.nan.
                if is_numeric_dtype(X[col]) and X[col].isna().any():
                    X[col] = X[col].astype(np.float64)

                # Cast pandas dtypes to numpy dtypes for earlier versions of sklearn.
                # Categorical dtypes don't need to be casted.
                # Note that 'is_category_dtype' is deprecated.
                if (
                    is_extension_array_dtype(X[col])
                    and not isinstance(X[col].dtype, CategoricalDtype)
                    and not is_datetime64_any_dtype(X[col])
                ):
                    dtype = X[col].dtype.type
                    X[col] = X[col].astype(dtype)

                    # When converting string to object, <NA> values becomes '<NA>'
                    # so we need to replace false missing values once more.
                    X[col] = _replace_missing_indicators(X[col])

                # For object dtype columns, convert to string to avoid mixed types.
                if is_object_dtype(X[col]):
                    mask = X[col].notnull()
                    X.loc[mask, col] = X.loc[mask, col].astype(str)

        if reset:
            X = to_datetime(X)
            X = _to_numeric(X)
            self.inferred_column_types_ = X.dtypes.to_dict()

        else:
            category_columns = X.select_dtypes("category").columns
            for col in category_columns:
                dtype = self.inferred_column_types_[col]
                dtype = _union_category(X[col], dtype)
                self.inferred_column_types_[col] = dtype
                X[col] = X[col].astype(dtype)

            # Enforce dtypes conversion using the dtypes seen during fit.
            # As this behavior is more aggressive than skrub's to_datetime
            # or _to_numeric and only makes sense for the TableVectorizer, we
            # define it here rather than within these two functions.
            # See: https://github.com/skrub-data/skrub/issues/837
            for column, dtype in self.inferred_column_types_.items():
                if is_numeric_dtype(dtype):
                    X[column] = pd.to_numeric(X[column], errors="coerce")

                elif is_datetime64_any_dtype(dtype):
                    X[column] = pd.to_datetime(X[column], errors="coerce")

                else:
                    X[column] = X[column].astype(dtype, errors="ignore")

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

        # Check for duplicate column names.
        duplicate_columns = {k for k, v in Counter(X.columns).items() if v > 1}
        if len(duplicate_columns) > 0:
            raise AssertionError(
                f"Duplicate column names in the dataframe: {duplicate_columns}"
            )

        # Check Pandas sparse arrays
        sparse_cols = [
            col for col in X.columns if isinstance(X[col].dtype, pd.SparseDtype)
        ]
        if len(sparse_cols) > 0:
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
        X : dataframe of shape (n_samples, n_features)
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

        In practice, it:
        - Converts features to their best possible types for scikit-learn estimators
          if ``auto_cast=True`` (see ``auto_cast`` docstring).
        - Classify columns based on their data types and match them to each
          dtype-specific transformers.
        - Use scikit-learn ColumnTransformer to run fit_transform on all transformers.

        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
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
        self._clone_transformers()

        self._check_feature_names(X, reset=True)
        X = self._check_X(X)
        self._check_n_features(X, reset=True)

        X = self._auto_cast(X, reset=True)

        # Filter ``X`` to keep only the columns that are not specified
        # explicitly by the user.
        X_filtered = X.drop(
            columns=[
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
        low_cardinality_columns, high_cardinality_columns = [], []
        for col in categorical_columns:
            if X[col].nunique() < self.cardinality_threshold:
                low_cardinality_columns.append(col)
            else:
                high_cardinality_columns.append(col)

        all_transformers = [
            ("numeric", self.numerical_transformer_, numeric_columns),
            ("datetime", self.datetime_transformer_, datetime_columns),
            (
                "low_cardinality",
                self.low_cardinality_transformer_,
                low_cardinality_columns,
            ),
            (
                "high_cardinality",
                self.high_cardinality_transformer_,
                high_cardinality_columns,
            ),
            *self.specific_transformers_,
        ]
        # Filter this list, by keeping only transformers with:
        # - at least one column
        # - a valid encoder or string (filter out if None)
        transformers = []
        for name, transformer, columns in all_transformers:
            if len(columns) > 0 and transformer is not None:
                transformers.append((name, transformer, columns))

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
        """Transform ``X`` by applying the fitted transformers on the columns.

        Parameters
        ----------
        X : dataframe of shape (n_samples, n_features)
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

        X = self._auto_cast(X, reset=False)

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
