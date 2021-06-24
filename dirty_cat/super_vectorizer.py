"""

This class implements the SuperVectorizer, which is a preprocessor used to
automatically apply encoders to different types of data, without the need to
manually categorize them beforehand, or construct complex Pipelines.

"""

# Author: Lilian Boulard <lilian@boulard.fr> | https://github.com/LilianBoulard

import numpy as np
import pandas as pd

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
            array.fillna(0, inplace=True)
        else:
            array.fillna('', inplace=True)
        array = array.reset_index(drop=True)
        return array
    elif isinstance(array, pd.DataFrame):
        # Replace missing values for pandas
        for col in array.columns:
            dtype_name = array[col].dtype.name
            if dtype_name == 'category' \
                    and ('' not in array[col].cat.categories):
                array[col].cat.add_categories('', inplace=True)
            array[col].replace(to_replace='?', value='', inplace=True)
            if dtype_name.startswith('int') or dtype_name.startswith('float'):
                array[col].fillna(0, inplace=True)
            else:
                array[col].fillna('', inplace=True)
        array = array.reset_index(drop=True)
        return array
    elif isinstance(array, np.ndarray):
        # Replace missing values for numpy
        array[np.where(np.isnan(array))] = 0
        return array
    else:
        raise ValueError(_ERR_MSG_UNSUPPORTED_ARR_T.format(type(array)))


class SuperVectorizer(ColumnTransformer):
    """
    Applies transformers to columns of an array depending
    on the characteristics of each column.
    Under the hood, it is an interface for scikit-learn's `ColumnTransformer`.

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

    """

    # Override required parameters
    _required_parameters = []

    def __init__(self, *,
                 cardinality_threshold: int = 20,
                 low_card_str_transformer: BaseEstimator = OneHotEncoder(),
                 high_card_str_transformer: BaseEstimator = SimilarityEncoder(),
                 low_card_cat_transformer: BaseEstimator = OneHotEncoder(),
                 high_card_cat_transformer: BaseEstimator = SimilarityEncoder(),
                 numerical_transformer: BaseEstimator = None,
                 datetime_transformer: BaseEstimator = None,
                 auto_cast: bool = False,
                 # Following parameters are inherited from ColumnTransformer
                 handle_missing: str = '',
                 remainder='passthrough',
                 sparse_threshold=0.3,
                 n_jobs=None,
                 transformer_weights=None,
                 verbose=False,
                 **kwargs
                 ):

        if 'transformers' in kwargs:
            raise ValueError(
                "Keyword argument 'transformers' is forbidden. "
                "If you already have your transformers ready, please use "
                "sklearn's ColumnTransformer directly."
            )

        super().__init__(
            transformers=[],
            remainder=remainder,
            sparse_threshold=sparse_threshold,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose,
            **kwargs
        )
        self.cardinality_threshold = cardinality_threshold
        self.low_card_str_transformer = low_card_str_transformer
        self.high_card_str_transformer = high_card_str_transformer
        self.low_card_cat_transformer = low_card_cat_transformer
        self.high_card_cat_transformer = high_card_cat_transformer
        self.numerical_transformer = numerical_transformer
        self.datetime_transformer = datetime_transformer
        self.auto_cast = auto_cast
        self.handle_missing = handle_missing

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

        def cast_astype(col):
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

        def cast_series(sr: pd.Series) -> pd.Series:
            if _has_missing_values(sr):
                raise ValueError(_ERR_MSG_FOUND_NAN)

            return cast_astype(sr)

        def cast_dataframe(df: pd.DataFrame) -> pd.DataFrame:
            return df.convert_dtypes()

        def cast_array(arr: np.array) -> np.array:

            def cast_column(col: np.array):
                """
                Method to call on each column,
                which will try to cast said column to the best possible type.
                """

                # Check if the array contains NaN or INF
                if _has_missing_values(col):
                    raise ValueError(_ERR_MSG_FOUND_NAN)
                if any(col.isinf()):
                    raise ValueError(_ERR_MSG_FOUND_INF)

                return cast_astype(col)

            try:
                # nD array
                arr = np.apply_along_axis(func1d=cast_column, axis=1, arr=arr)
            except np.AxisError:
                # 1D array
                arr = cast_column(arr)

            return arr

        if isinstance(X, pd.Series):
            return cast_series(X)
        elif isinstance(X, pd.DataFrame):
            return cast_dataframe(X)
        elif isinstance(X, np.ndarray):
            return cast_array(X)

    def _transform(self, X) -> pd.DataFrame:
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
            X = self._auto_cast_array(X)

        return X

    def transform(self, X):
        X = self._transform(X)
        # [Black magic] - Calls the overridden method `transform`.
        return ColumnTransformer.transform(self, X)

    def fit_transform(self, X, y=None):
        X = self._transform(X)

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

        # Construct the transformers
        transformers_df = pd.DataFrame([
            ['numeric', self.numerical_transformer, numeric_columns],
            ['datetime', self.datetime_transformer, datetime_columns],
            ['low_card_str', self.low_card_str_transformer, low_card_str_columns],
            ['high_card_str', self.high_card_str_transformer, high_card_str_columns],
            ['low_card_cat', self.low_card_cat_transformer, low_card_cat_columns],
            ['high_card_cat', self.high_card_cat_transformer, high_card_cat_columns],
        ], columns=['name', 'transformer', 'columns'])
        # Get mask of lines with valid encoders.
        # True: valid encoders
        valid_encoders_mask = np.invert(transformers_df['transformer'].isnull())
        # Get mask of lines with non-empty columns
        # True: non-empty columns field
        valid_columns_mask = transformers_df['columns'].apply(lambda x: True if x != [] else False)

        mask = valid_encoders_mask & valid_columns_mask
        self.transformers = transformers_df[mask].values.tolist()

        if not self.transformers:
            raise RuntimeError('No transformers could be generated !')

        if self.verbose:
            print(f'[SuperVectorizer] Assigned transformers: {self.transformers}')

        # [Black magic] - Calls the overridden method `fit_transform`.
        return ColumnTransformer.fit_transform(self, X, y)
