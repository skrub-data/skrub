"""

This class implements the SuperVectorizer, which is a preprocessor used to
automatically apply encoders to different types of data, without the need to
manually categorize them beforehand, or construct complex Pipelines.

"""

# Author: Lilian Boulard <lilian@boulard.fr> | https://github.com/LilianBoulard

import sklearn

import numpy as np
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


def _replace_missing_in_col(df: pd.Series, value: str = "missing") -> pd.Series:
    """
    Takes a Series with string data, replaces the missing values, and returns it.
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

    cardinality_threshold: int, default=40
        Two lists of features will be created depending on this value: strictly
        under this value, the low cardinality categorical values, and above or
        equal, the high cardinality categorical values.
        Different encoders will be applied to these two groups, defined by
        the parameters `low_card_cat_transformer` and
        `high_card_cat_transformer` respectively.

    low_card_cat_transformer: Transformer or str or None, default=OneHotEncoder()
        Transformer used on categorical/string features with low cardinality
        (threshold is defined by `cardinality_threshold`).
        Can either be a transformer object instance (e.g. `OneHotEncoder()`),
        a `Pipeline` containing the preprocessing steps,
        None to apply `remainder`, 'drop' for dropping the columns,
        or 'passthrough' to return the unencoded columns.

    high_card_cat_transformer: Transformer or str or None, default=GapEncoder(n_components=30)
        Transformer used on categorical/string features with high cardinality
        (threshold is defined by `cardinality_threshold`).
        Can either be a transformer object instance (e.g. `GapEncoder()`),
        a `Pipeline` containing the preprocessing steps,
        None to apply `remainder`, 'drop' for dropping the columns,
        or 'passthrough' to return the unencoded columns.

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

    impute_missing: str, default='auto'
        When to impute missing values in string columns.
        'auto' will impute missing values if it's considered appropriate
        (we are using an encoder that does not support missing values and/or
        specific versions of pandas, numpy and scikit-learn).
        'force' will impute all missing values.
        'skip' will not impute at all.
        When imputed, missing values are replaced by the string 'missing'.
        See also attribute `imputed_columns_`.
        
    remainder : {'drop', 'passthrough'} or estimator, default='drop'
        By default, only the specified columns in `transformers` are
        transformed and combined in the output, and the non-specified
        columns are dropped. (default of ``'drop'``).
        By specifying ``remainder='passthrough'``, all remaining columns that
        were not specified in `transformers` will be automatically passed
        through. This subset of columns is concatenated with the output of
        the transformers.
        By setting ``remainder`` to be an estimator, the remaining
        non-specified columns will use the ``remainder`` estimator. The
        estimator must support :term:`fit` and :term:`transform`.
        Note that using this feature requires that the DataFrame columns
        input at :term:`fit` and :term:`transform` have identical order.
        
    sparse_threshold: float, default=0.3
        If the output of the different transformers contains sparse matrices,
        these will be stacked as a sparse matrix if the overall density is
        lower than this value. Use sparse_threshold=0 to always return dense. 
        When the transformed output consists of all dense data, the stacked result
        will be dense, and this keyword will be ignored.
        
    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
        
    transformer_weights : dict, default=None
        Multiplicative weights for features per transformer. The output of the
        transformer is multiplied by these weights. Keys are transformer names,
        values the weights.
        
    verbose : bool, default=False
        If True, the time elapsed while fitting each transformer will be
        printed as it is completed

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

    types_: Dict[str, type]
        A mapping of inferred types per column.
        Key is the column name, value is the inferred dtype.

    imputed_columns_: List[str]
        The list of columns in which we imputed the missing values.

    """

    # Override required parameters
    _required_parameters = []
    OptionalEstimator = Optional[Union[BaseEstimator, str]]

    def __init__(self, *,
                 cardinality_threshold: int = 40,
                 low_card_cat_transformer: Optional[Union[BaseEstimator, str]] = OneHotEncoder(),
                 high_card_cat_transformer: Optional[Union[BaseEstimator, str]] = GapEncoder(n_components=30),
                 numerical_transformer: Optional[Union[BaseEstimator, str]] = None,
                 datetime_transformer: Optional[Union[BaseEstimator, str]] = None,
                 auto_cast: bool = True,
                 impute_missing: str = 'auto',
                 # Following parameters are inherited from ColumnTransformer
                 remainder: str = 'passthrough',
                 sparse_threshold: float = 0.3,
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

    @staticmethod
    def _auto_cast(X: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a dataframe and tries to convert its columns to the best
        possible data type.

        Parameters
        ----------
        X : {dataframe} of shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        pd.DataFrame
            The same pandas DataFrame, with its columns casted to the best possible
            data type.
        """
        from pandas.core.dtypes.base import ExtensionDtype

        # Handle missing values
        for col in X.columns:
            contains_missing: bool = _has_missing_values(X[col])
            # Convert pandas' NaN value (pd.NA) to numpy NaN value (np.nan)
            # because the former tends to raise all kind of issues when dealing
            # with scikit-learn (as of version 0.24).
            if contains_missing:
                # Some numerical dtypes like Int64 or Float64 only support
                # pd.NA so they must be converted to np.float64 before.
                if pd.api.types.is_numeric_dtype(X[col]):
                    X[col] = X[col].astype(np.float64)
                X[col].fillna(value=np.nan, inplace=True)
        STR_NA_VALUES = ['null', '', '1.#QNAN', '#NA', 'nan', '#N/A N/A', '-1.#QNAN', '<NA>', '-1.#IND', '-nan', 'n/a',
                         '-NaN', '1.#IND', 'NULL', 'NA', 'N/A', '#N/A', 'NaN']  # taken from pandas.io.parsers (version 1.1.4)
        X = X.replace(STR_NA_VALUES + [None, "?", "..."],
                      np.nan)
        X = X.replace(r'^\s+$', np.nan, regex=True) # replace whitespace only

        # Convert to best possible data type
        for col in X.columns:
            if not pd.api.types.is_datetime64_any_dtype(X[col]): # we don't want to cast datetime64
                try:
                    X[col] = pd.to_numeric(X[col], errors='raise')
                except:
                    # Only try to convert to datetime if the variable isn't numeric.
                    try:
                        X[col] = pd.to_datetime(X[col], errors='raise')
                    except:
                        pass
            # Cast pandas dtypes to numpy dtypes
            # for earlier versions of sklearn
            if issubclass(X[col].dtype.__class__, ExtensionDtype):
                try:
                    X[col] = X[col].astype(X[col].dtype.type, errors='ignore')
                except (TypeError, ValueError):
                    pass
        return X

    def transform(self, X) -> np.ndarray:
        """Transform X by applying transformers on each column, then concatenate results.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        X_t : {array-like, sparse matrix} of \
                shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers. If
            any result is a sparse matrix, everything will be converted to
            sparse matrices.

        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        else:
            # Create a copy to avoid altering the original data.
            X = X.copy()
        # Auto cast the imported data to avoid having issues with
        # `object` dtype when we will cast columns to the fitted types.
        if self.auto_cast:
            X = self._auto_cast(X)
            self.types_ = {c: t for c, t in zip(X.columns, X.dtypes)}
        if X.shape[1] != len(self.columns_):
            raise ValueError("Passed array does not match column count of "
                             f"array seen at fit time. Got {X.shape[1]} "
                             f"columns, expected {len(self.columns_)}")

        # If the DataFrame does not have named columns already,
        # apply the learnt columns
        if pd.api.types.is_numeric_dtype(X.columns):
            X.columns = self.columns_

        for col in self.imputed_columns_:
            X[col] = _replace_missing_in_col(X[col])

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
        # Convert to pandas DataFrame if not already.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        else:
            # Create a copy to avoid altering the original data.
            X = X.copy()

        self.columns_ = X.columns
        # If auto_cast is True, we'll find and apply the best possible type
        # to each column.
        # We'll keep the results so we can apply the types in transform.
        if self.auto_cast:
            X = self._auto_cast(X)
            self.types_ = {c: t for c, t in zip(X.columns, X.dtypes)}

        # Select columns by dtype
        numeric_columns = X.select_dtypes(include=['int', 'float']).columns.to_list()
        categorical_columns = X.select_dtypes(include=['string', 'object', 'category']).columns.to_list()
        datetime_columns = X.select_dtypes(include='datetime').columns.to_list()

        # Divide categorical columns by cardinality
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

        self.imputed_columns_ = []
        if _has_missing_values(X):

            if self.impute_missing == 'force':
                for col in X.columns:
                    # Do not impute numeric columns
                    if not pd.api.types.is_numeric_dtype(X[col]):
                        X[col] = _replace_missing_in_col(X[col])
                        self.imputed_columns_.append(col)

            elif self.impute_missing == 'skip':
                pass

            elif self.impute_missing == 'auto':
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
                        for col in cols:
                            # Do not impute numeric columns
                            if not pd.api.types.is_numeric_dtype(X[col]):
                                X[col] = _replace_missing_in_col(X[col])
                                self.imputed_columns_.append(col)

            else:
                raise ValueError(
                    "Invalid value for `impute_missing`, expected any of "
                    f"{'auto', 'force', 'skip'}, got {self.impute_missing!r}."
                )

        # If there was missing values imputation, we cast the DataFrame again,
        # as pandas give different types depending whether a column has
        # missing values or not.
        if self.imputed_columns_:
            X = self._auto_cast(X)

        if self.verbose:
            print(f'[SuperVectorizer] Assigned transformers: {self.transformers}')

        return super().fit_transform(X, y)

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """
        Returns clean feature names with format
        "<column_name>_<value>" if encoded by OneHotEncoder or alike,
        e.g. "job_title_Police officer",
        or "<column_name>" if not encoded.
        """
        if _sklearn_loose_version < LooseVersion('0.23'):
            try:
                if _sklearn_loose_version < LooseVersion('1.0'):
                    ct_feature_names = super().get_feature_names()
                else:
                    ct_feature_names = super().get_feature_names_out()
            except NotImplementedError:
                raise NotImplementedError(
                    'Prior to sklearn 0.23, get_feature_names with '
                    '"passthrough" is unsupported. To use the method, '
                    'either make sure there is no "passthrough" in the '
                    'transformers, or update your copy of scikit-learn.'
                )
        else:
            if _sklearn_loose_version < LooseVersion('1.0'):
                ct_feature_names = super().get_feature_names()
            else:
                ct_feature_names = super().get_feature_names_out()
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
    
    def get_feature_names(self) -> List[str]:
        """ Deprecated, use "get_feature_names_out"
        """
        warn(
            "get_feature_names is deprecated in scikit-learn > 1.0. "
            "use get_feature_names_out instead",
            DeprecationWarning,
            )
        return self.get_feature_names_out()
