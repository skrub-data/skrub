"""

This class implements the SuperVectorizer, which is a preprocessor used to
automatically apply encoders to different types of data, without the need to
manually categorize them beforehand, or construct complex Pipelines.

"""

# Author: Lilian Boulard <lilian@boulard.fr> | https://github.com/LilianBoulard

import sklearn

import pandas as pd

from warnings import warn
from typing import Union, Optional, List
from distutils.version import LooseVersion

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from dirty_cat import GapEncoder


_sklearn_loose_version = LooseVersion(sklearn.__version__)


def _has_missing_values(df: Union[pd.DataFrame, pd.Series]) -> bool:
    """
    Returns True if `array` contains missing values, False otherwise.
    """
    return any(df.isnull())


def _replace_missing_col(df: pd.Series, value: str = "missing") -> pd.Series:
    """
    Takes a Series, replaces the missing values, and returns it.
    """
    dtype_name = df.dtype.name
    if dtype_name == 'category' and (value not in df.cat.categories):
        df = df.cat.add_categories(value)
    df = df.fillna(value=value)
    return df


class SuperVectorizer(ColumnTransformer):
    """
    Easily transforms a heterogeneous data table (such as a dataframe) to
    a numerical array for machine learning. For this it transforms each
    column depending on its data type.
    It provides a simplified interface for scikit-learn's `ColumnTransformer`.

    .. versionadded:: 0.2.0

    Parameters
    ----------

    cardinality_threshold: int, default=20
        Two lists of features will be created depending on this value: strictly
        under this value, the low cardinality categorical values, and above or
        equal, the high cardinality categorical values.
        Different encoders will be applied to these two groups, defined by
        the parameters `low_card_str_transformer`/`low_card_cat_transformer` and
        `high_card_str_transformer`/`high_card_cat_transformer` respectively.

    low_card_str_transformer: Transformer or str or None, default=OneHotEncoder()
        Transformer used on features with low cardinality (threshold is
        defined by `cardinality_threshold`).
        Can either be a transformer object instance (e.g. `OneHotEncoder()`),
        a `Pipeline` containing the preprocessing steps,
        None to apply `remainder`, 'drop' for dropping the columns,
        or 'passthrough' to return the unencoded columns.

    high_card_str_transformer: Transformer or str or None, default=GapEncoder()
        Transformer used on features with high cardinality (threshold is
        defined by `cardinality_threshold`).
        Can either be a transformer object instance (e.g. `GapEncoder()`),
        a `Pipeline` containing the preprocessing steps,
        None to apply `remainder`, 'drop' for dropping the columns,
        or 'passthrough' to return the unencoded columns.

    low_card_cat_transformer: Transformer or str or None, default=OneHotEncoder()
        Same as `low_card_str_transformer`.

    high_card_cat_transformer: Transformer or str or None, default=GapEncoder()
        Same as `high_card_str_transformer`.

    numerical_transformer: Transformer or str or None, default=None
        Transformer used on numerical features.
        Can either be a transformer object instance (e.g. `StandardScaler()`),
        a `Pipeline` containing the preprocessing steps,
        None to apply `remainder`, 'drop' for dropping the columns,
        or 'passthrough' to return the unencoded columns.

    datetime_transformer: Transformer or str or None, default=None
        Transformer used on datetime features.
        Can either be a transformer object instance,
        a `Pipeline` containing the preprocessing steps,
        None to apply `remainder`, 'drop' for dropping the columns,
        or 'passthrough' to return the unencoded columns.

    auto_cast: bool, default=True
        If set to `True`, will try to convert each column to the best possible
        data type (dtype).

    handle_missing: str, default=''
        One of the following values: 'error' or '' (empty).
        Defines how the encoder will handle missing values.
        If set to 'error', will raise ValueError.
        If set to '', will impute the missing values (pd.NA) with blank strings.

    Attributes
    ----------

    transformers_: List[Tuple[str, Union[str, BaseEstimator], Union[str, int]]]
        The final distribution of columns.
        List of three-tuple containing
        (1) the name of the category
        (2) the encoder/transformer instance which will be applied
        or "passthrough" or "drop"
        (3) the list of column names or index

    columns_: List[Union[str, int]]
        The column names of fitted array.

    self.types_: Dict[int, type]
        A learnt mapping of type by column.
        Key is the column name, value is the final dtype.

    self.imputed_columns_: List[str]
        A list of columns that had missing values and were imputed.

    """

    # Override required parameters
    _required_parameters = []
    OptionalEstimator = Optional[Union[BaseEstimator, str]]

    def __init__(self, *,
                 cardinality_threshold: int = 20,
                 low_card_str_transformer: OptionalEstimator = OneHotEncoder(handle_unknown='ignore'),
                 high_card_str_transformer: OptionalEstimator = GapEncoder(),
                 low_card_cat_transformer: OptionalEstimator = OneHotEncoder(handle_unknown='ignore'),
                 high_card_cat_transformer: OptionalEstimator = GapEncoder(),
                 numerical_transformer: OptionalEstimator = None,
                 datetime_transformer: OptionalEstimator = None,
                 auto_cast: bool = True,
                 impute_missing: str = 'auto',
                 # Following parameters are inherited from ColumnTransformer
                 handle_missing: str = '',
                 remainder: str = 'passthrough',
                 sparse_threshold: float = 0.3,
                 n_jobs: int = None,
                 transformer_weights=None,
                 verbose: bool = False,
                 ):
        super().__init__(transformers=[])

        self.cardinality_threshold = cardinality_threshold
        self.low_card_str_transformer = low_card_str_transformer
        self.high_card_str_transformer = high_card_str_transformer
        self.low_card_cat_transformer = low_card_cat_transformer
        self.high_card_cat_transformer = high_card_cat_transformer
        self.numerical_transformer = numerical_transformer
        self.datetime_transformer = datetime_transformer
        self.auto_cast = auto_cast
        self.impute_missing = impute_missing

        self.handle_missing = handle_missing
        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose

    def _auto_cast(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a pandas DataFrame and tries to convert its columns to the best
        possible data type.

        Parameters
        ----------
        X: pd.DataFrame
            Input data as a pandas DataFrame.

        Returns
        -------
        array
            The same array, with its columns casted to the best possible
            data type.
            If there are missing values in a column, it won't change.
        """
        for col in X.columns:
            if not _has_missing_values(X[col]):
                X[col] = X[col].convert_dtypes()

        # Cast pandas dtypes to numpy dtypes
        # for earlier versions of sklearn
        from pandas.core.dtypes.base import ExtensionDtype
        for column in X:
            dtype = X[column].dtype
            if issubclass(dtype.__class__, ExtensionDtype):
                try:
                    X[column] = X[column].astype(dtype.type)
                except TypeError:
                    pass
        return X

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            The data to be transformed by subset.

        Returns
        -------
        X_t : {array-like, sparse matrix} of \
                shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers. If
            any result is a sparse matrix, everything will be converted to
            sparse matrices.

        """
        # Create a copy to avoid altering the original data.
        X = X.copy()
        # Convert to pandas DataFrame if not already.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            # Check the number of columns matches the fitted array's.
            if X.shape[1] != len(self.columns_):
                raise ValueError("Passed array does not match column count of "
                                 f"fitter array. Got X.shape={X.shape}")
            # If the DataFrame does not have named columns already,
            # apply the learnt columns
            if isinstance(X.columns, pd.RangeIndex):
                X.columns = self.columns_

        if self.auto_cast:
            X.columns = self.columns_
            X = X.astype(self.types_)

        for col in self.imputed_columns_:
            X[col] = _replace_missing_col(X[col])

        return super().transform(X)

    def fit_transform(self, X, y=None):
        """
        Fit all transformers, transform the data, and concatenate results.

        Parameters
        ----------
        X: {array-like, dataframe} of shape (n_samples, n_features)
            Input data, of which specified subsets are used to fit the
            transformers.

        y: array-like of shape (n_samples,), default=None
            Targets for supervised learning.

        Returns
        -------
        X_t: {array-like, sparse matrix} of shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers. If
            any result is a sparse matrix, everything will be converted to
            sparse matrices.

        Raises
        ------
        RuntimeError
            If no transformers could be constructed,
            usually because transformers passed do not match any column.
            To fix the issue, try passing the least amount of None as encoders.
        """
        # Create a copy to avoid altering the original data.
        X = X.copy()
        # Convert to pandas DataFrame if not already.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # X = self._transform(X)
        self.columns_ = X.columns
        # If auto_cast is True, we'll find and apply the best possible type
        # to each column.
        # We'll keep the results so we can apply the types in transform.
        if self.auto_cast:
            X = self._auto_cast(X)
            self.types_ = {c: t for c, t in zip(X.columns, X.dtypes)}

        # Select columns by dtype
        numeric_columns = X.select_dtypes(include=['int', 'float']).columns.to_list()
        string_columns = X.select_dtypes(include=['string', 'object']).columns.to_list()
        categorical_columns = X.select_dtypes(include='category').columns.to_list()
        datetime_columns = X.select_dtypes(include='datetime').columns.to_list()

        # Divide string and categorical columns by cardinality
        low_card_str_columns = [
            col for col in string_columns
            if X[col].nunique() < self.cardinality_threshold
        ]
        high_card_str_columns = [
            col for col in string_columns
            if X[col].nunique() >= self.cardinality_threshold
        ]
        low_card_cat_columns = [
            col for col in categorical_columns
            if X[col].nunique() < self.cardinality_threshold
        ]
        high_card_cat_columns = [
            col for col in categorical_columns
            if X[col].nunique() >= self.cardinality_threshold
        ]

        # Next part: construct the transformers
        # Create the list of all the transformers.
        all_transformers = [
            ('numeric', self.numerical_transformer, numeric_columns),
            ('datetime', self.datetime_transformer, datetime_columns),
            ('low_card_str', self.low_card_str_transformer, low_card_str_columns),
            ('high_card_str', self.high_card_str_transformer, high_card_str_columns),
            ('low_card_cat', self.low_card_cat_transformer, low_card_cat_columns),
            ('high_card_cat', self.high_card_cat_transformer, high_card_cat_columns),
        ]
        # We will now filter this list, by keeping only the ones with:
        # - at least one column
        # - a valid encoder or string (filter out if None)
        self.transformers = []
        for trans in all_transformers:
            name, enc, cols = trans  # Unpack
            if len(cols) > 0 and enc is not None:
                self.transformers.append(trans)

        if len(self.transformers) == 0:
            raise RuntimeError('No transformers could be generated !')

        # Replace missing values in specific cases
        if _has_missing_values(X):
            self.imputed_columns_ = []
            if self.impute_missing == 'auto':
                for name, trans, cols in all_transformers:
                    # At each iteration, we'll manipulate a boolean,
                    # and depending on its value at the end of the loop,
                    # we will or will not replace the missing values in
                    # the columns.
                    impute: bool = False

                    if isinstance(trans, OneHotEncoder) \
                            and _sklearn_loose_version < LooseVersion('0.24'):
                        impute = True

                    if impute:
                        self.imputed_columns_.extend(cols)
                        for col in cols:
                            X[col] = _replace_missing_col(X[col])

        # If there was missing values imputation, we cast the DataFrame again,
        # as pandas give different types depending whether a column has
        # missing values or not.
        if self.imputed_columns_:
            X = self._auto_cast(X)

        if self.verbose:
            print(f'[SuperVectorizer] Assigned transformers: {self.transformers}')

        return super().fit_transform(X, y)

    def get_feature_names(self) -> List[str]:
        """
        Returns clean feature names with format
        "<column_name>_<value>" if encoded by OneHotEncoder or alike,
        e.g. "job_title_Police officer",
        or "<column_name>" if not encoded.
        """
        if _sklearn_loose_version < LooseVersion('0.23'):
            try:
                ct_feature_names = super().get_feature_names()
            except NotImplementedError:
                raise NotImplementedError(
                    'Prior to sklearn 0.23, get_feature_names with '
                    '"passthrough" is unsupported. To use the method, '
                    'either make sure there is no "passthrough" in the '
                    'transformers, or update your copy of scikit-learn.'
                )
        else:
            ct_feature_names = super().get_feature_names()
        all_trans_feature_names = []

        for name, trans, cols, _ in self._iter(fitted=True):
            if isinstance(trans, str):
                if trans == 'drop':
                    continue
                elif trans == 'passthrough':
                    if all(isinstance(col, int) for col in cols):
                        cols = [self.columns_[i] for i in cols]
                    all_trans_feature_names.extend(cols)
                continue
            if not hasattr(trans, 'get_feature_names'):
                all_trans_feature_names.extend(cols)
            else:
                trans_feature_names = trans.get_feature_names(cols)
                all_trans_feature_names.extend(trans_feature_names)

        if len(ct_feature_names) != len(all_trans_feature_names):
            warn('Could not extract clean feature names ; returning defaults.')
            return ct_feature_names

        return all_trans_feature_names
