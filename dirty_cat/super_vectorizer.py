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

from dirty_cat import SimilarityEncoder


_ERR_MSG_UNSUPPORTED_ARR_T = 'Unsupported array type: {}'
_ERR_MSG_FOUND_NAN = 'Found NaN in array'
_ERR_MSG_FOUND_INF = 'Found INF in array'


def _has_missing_values(array) -> bool:
    """
    Returns True if `array` contains missing values, False otherwise.
    """
    if isinstance(array, pd.DataFrame) or isinstance(array, pd.Series):
        return any(array.isnull())
    elif isinstance(array, np.ndarray):
        return np.isnan(array).any()
    else:
        raise ValueError(_ERR_MSG_UNSUPPORTED_ARR_T.format(type(array)))


def _replace_missing(array):
    """
    Takes an array, replaces the missing values with a specific value, and returns it.
    """
    if isinstance(array, pd.Series):
        dtype_name = array.dtype.name
        if dtype_name == 'category' \
                and ('' not in array.cat.categories):
            array = array.cat.add_categories('')
        if dtype_name.startswith('int') or dtype_name.startswith('float'):
            array = array.fillna(0)
        else:
            array = array.fillna('')
        array = array.reset_index(drop=True)
        return array
    elif isinstance(array, pd.DataFrame):
        # Replace missing values for pandas
        for col in array.columns:
            dtype_name = array[col].dtype.name
            if dtype_name == 'category' \
                    and ('' not in array[col].cat.categories):
                array[col] = array[col].cat.add_categories('')
            array[col] = array[col].replace(to_replace='?', value='')
            if dtype_name.startswith('int') or dtype_name.startswith('float'):
                array[col] = array[col].fillna(0)
            else:
                array[col] = array[col].fillna('')
        array = array.reset_index(drop=True)
        return array
    elif isinstance(array, np.ndarray):
        # Replace missing values for numpy
        # Warning: changes the array in-place.
        # TODO: Add user warning regarding this issue,
        # or find a fix.
        array[np.where(np.isnan(array))] = 0
        return array
    else:
        raise ValueError(_ERR_MSG_UNSUPPORTED_ARR_T.format(type(array)))


class SuperVectorizer(ColumnTransformer):
    """
    Applies transformers to columns of an array depending
    on the characteristics of each column.
    Under the hood, it is an interface for scikit-learn's `ColumnTransformer`.

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

    low_card_str_transformer, default=OneHotEncoder()
        Transformer used on features with low cardinality (threshold is
        defined by `cardinality_threshold`).
        Can either be a transformer object instance (e.g. `OneHotEncoder()`),
        a `Pipeline` containing the preprocessing steps,
        None to apply `remainder`, 'drop' for dropping the columns,
        or 'passthrough' to return the unencoded columns.

    high_card_str_transformer, default=GapEncoder()
        Transformer used on features with high cardinality (threshold is
        defined by `cardinality_threshold`).
        Can either be a transformer object instance (e.g. `GapEncoder()`),
        a `Pipeline` containing the preprocessing steps,
        None to apply `remainder`, 'drop' for dropping the columns,
        or 'passthrough' to return the unencoded columns.

    low_card_cat_transformer, default=OneHotEncoder()
        Same as `low_card_str_transformer`.

    high_card_cat_transformer, default=GapEncoder()
        Same as `high_card_str_transformer`.

    numerical_transformer, default=None
        Transformer used on numerical features.
        Can either be a transformer object instance (e.g. `StandardScaler()`),
        a `Pipeline` containing the preprocessing steps,
        None to apply `remainder`, 'drop' for dropping the columns,
        or 'passthrough' to return the unencoded columns.

    datetime_transformer, default=None
        Transformer used on datetime features.
        Can either be a transformer object instance,
        a `Pipeline` containing the preprocessing steps,
        None to apply `remainder`, 'drop' for dropping the columns,
        or 'passthrough' to return the unencoded columns.

    auto_cast: bool, default=False
        If set to `True`, will try to convert each column to the best possible
        data type (dtype).
        Experimental. It is advised to cast them beforehand, and leave this false.

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

    """

    # Override required parameters
    _required_parameters = []

    def __init__(self, *,
                 cardinality_threshold: int = 20,
                 low_card_str_transformer: Optional[Union[BaseEstimator, str]] = OneHotEncoder(),
                 high_card_str_transformer: Optional[Union[BaseEstimator, str]] = SimilarityEncoder(),
                 low_card_cat_transformer: Optional[Union[BaseEstimator, str]] = OneHotEncoder(),
                 high_card_cat_transformer: Optional[Union[BaseEstimator, str]] = SimilarityEncoder(),
                 numerical_transformer: Optional[Union[BaseEstimator, str]] = None,
                 datetime_transformer: Optional[Union[BaseEstimator, str]] = None,
                 auto_cast: bool = False,
                 # Following parameters are inherited from ColumnTransformer
                 handle_missing: str = '',
                 remainder='passthrough',
                 sparse_threshold=0.3,
                 n_jobs=None,
                 transformer_weights=None,
                 verbose=False,
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
        self.handle_missing = handle_missing

        self.remainder = remainder
        self.sparse_threshold = sparse_threshold
        self.n_jobs = n_jobs
        self.transformer_weights = transformer_weights
        self.verbose = verbose

        self.columns_ = []

    @staticmethod
    def _cast_astype(col):
        # First, try to convert the column to floats
        try:
            return col.astype(float)
        except ValueError:
            # Couldn't cast
            pass

        # Next, to integers
        try:
            return col.astype(int)
        except ValueError:
            pass

        # Finally, to strings
        # We are not using try-except because this should work no matter the data.
        # (to be confirmed).
        return col.astype(str)

    @staticmethod
    def _cast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        return df.convert_dtypes()

    def _cast_column(self, col: np.array):
        """
        Method to call on each column,
        which will try to cast said column to the best possible type.
        """

        # Check if the array contains NaN or INF
        if _has_missing_values(col):
            raise ValueError(_ERR_MSG_FOUND_NAN)
        if any(col.isinf()):
            raise ValueError(_ERR_MSG_FOUND_INF)

        return self._cast_astype(col)

    def _cast_series(self, sr: pd.Series) -> pd.Series:
        if _has_missing_values(sr):
            raise ValueError(_ERR_MSG_FOUND_NAN)

        return self._cast_astype(sr)

    def _cast_array(self, arr: np.array) -> np.array:

        try:
            # nD array
            arr = np.apply_along_axis(func1d=self._cast_column, axis=1, arr=arr)
        except np.AxisError:
            # 1D array
            arr = self._cast_column(arr)

        return arr

    def _auto_cast_array(self, X):
        """
        Takes an array and tries to convert its columns to the best possible
        data type.

        Parameters
        ----------
        X: array-like
            Input data.

        Returns
        -------
        array
            The same array, with its columns casted to the best possible
            data type.

        Raises
        ------
        RuntimeError
            If no transformers could be constructed,
            usually because transformers passed do not match any column.
            To fix the issue, try passing the least amount of None as encoders.

        """

        if isinstance(X, pd.Series):
            return self._cast_series(X)
        elif isinstance(X, pd.DataFrame):
            return self._cast_dataframe(X)
        elif isinstance(X, np.ndarray):
            return self._cast_array(X)

    def _transform(self, X) -> pd.DataFrame:
        # Create a copy to avoid altering the original data.
        X = X.copy()
        # Convert to pandas DataFrame if not already.
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Detect if the array contains missing values.
        if _has_missing_values(X):
            if self.handle_missing == '':
                X = _replace_missing(X)
            elif self.handle_missing == 'error':
                raise ValueError('Array contains missing values')
            else:
                raise ValueError("Invalid 'handle_missing' value. "
                                 "Expected any of {'', 'error'}, "
                                 f"got {self.handle_missing}")

        if self.auto_cast:
            from pandas.core.dtypes.base import ExtensionDtype
            X = self._auto_cast_array(X)

            if LooseVersion(sklearn.__version__) <= LooseVersion('0.22'):
                # Cast pandas dtypes to numpy dtypes
                # for earlier versions of sklearn
                for column in X:
                    dtype = X[column].dtype
                    if issubclass(dtype.__class__, ExtensionDtype):
                        try:
                            X[column] = X[column].astype(dtype.type)
                        except TypeError:
                            pass

        return X

    def transform(self, X):
        X = self._transform(X)
        return super().transform(X)

    def fit_transform(self, X, y=None):
        X = self._transform(X)
        self.columns_ = X.columns

        # Select columns by dtype
        numeric_columns = X.select_dtypes(include=['int', 'float']).columns.to_list()
        string_columns = X.select_dtypes(include=['string', 'object']).columns.to_list()
        categorical_columns = X.select_dtypes(include='category').columns.to_list()
        datetime_columns = X.select_dtypes(include='datetime').columns.to_list()

        # Divide string and categorical columns by cardinality
        low_card_str_columns = [col for col in string_columns if X[col].nunique() < self.cardinality_threshold]
        high_card_str_columns = [col for col in string_columns if X[col].nunique() >= self.cardinality_threshold]
        low_card_cat_columns = [col for col in categorical_columns if X[col].nunique() < self.cardinality_threshold]
        high_card_cat_columns = [col for col in categorical_columns if X[col].nunique() >= self.cardinality_threshold]

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
