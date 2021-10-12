import numpy as np
import pytest
import sklearn
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from distutils.version import LooseVersion

from dirty_cat import SuperVectorizer
from dirty_cat import GapEncoder


def check_same_transformers(expected_transformers: dict, actual_transformers: list):
    # Construct the dict from the actual transformers
    actual_transformers_dict = dict([(name, cols) for name, trans, cols in actual_transformers])
    assert actual_transformers_dict == expected_transformers


def _get_clean_dataframe():
    """
    Creates a simple DataFrame with various types of data,
    and without missing values.
    """
    return pd.DataFrame({
        'int': pd.Series([15, 56, 63, 12, 44], dtype='int'),
        'float': pd.Series([5.2, 2.4, 6.2, 10.45, 9.], dtype='float'),
        'str1': pd.Series(['public', 'private', 'private', 'private', 'public'], dtype='string'),
        'str2': pd.Series(['officer', 'manager', 'lawyer', 'chef', 'teacher'], dtype='string'),
        'cat1': pd.Series(['yes', 'yes', 'no', 'yes', 'no'], dtype='category'),
        'cat2': pd.Series(['20K+', '40K+', '60K+', '30K+', '50K+'], dtype='category'),
    })


def _get_dirty_dataframe():
    """
    Creates a simple DataFrame with some missing values.
    We'll use different types of missing values (np.nan, pd.NA, None)
    to see how robust the vectorizer is.
    """
    return pd.DataFrame({
        'int': pd.Series([15, 56, pd.NA, 12, 44], dtype='Int64'),
        'float': pd.Series([5.2, 2.4, 6.2, 10.45, np.nan], dtype='Float64'),
        'str1': pd.Series(['public', np.nan, 'private', 'private', 'public'], dtype='object'),
        'str2': pd.Series(['officer', 'manager', None, 'chef', 'teacher'], dtype='object'),
        'cat1': pd.Series([np.nan, 'yes', 'no', 'yes', 'no'], dtype='object'),
        'cat2': pd.Series(['20K+', '40K+', '60K+', '30K+', np.nan], dtype='object'),
    })


def _test_possibilities(
        X,
        expected_transformers_df,
        expected_transformers_2,
        expected_transformers_np_no_cast,
        expected_transformers_series,
        expected_transformers_plain,
        expected_transformers_np_cast,
):
    """
    Do a bunch of tests with the SuperVectorizer.
    We take some expected transformers results as argument. They're usually
    lists or dictionaries.
    """
    # Test with low cardinality and a StandardScaler for the numeric columns
    vectorizer_base = SuperVectorizer(
        cardinality_threshold=4,
        # we must have n_samples = 5 >= n_components
        high_card_cat_transformer=GapEncoder(n_components=2),
        numerical_transformer=StandardScaler(),
    )
    # Warning: order-dependant
    vectorizer_base.fit_transform(X)
    check_same_transformers(expected_transformers_df, vectorizer_base.transformers)

    # Test with higher cardinality threshold and no numeric transformer
    vectorizer_default = SuperVectorizer()  # Using default values
    vectorizer_default.fit_transform(X)
    check_same_transformers(expected_transformers_2, vectorizer_default.transformers)

    # Test with a numpy array
    arr = X.to_numpy()
    # Instead of the columns names, we'll have the column indices.
    vectorizer_base.fit_transform(arr)
    check_same_transformers(expected_transformers_np_no_cast, vectorizer_base.transformers)

    # Test with pandas series
    vectorizer_base.fit_transform(X['cat1'])
    check_same_transformers(expected_transformers_series, vectorizer_base.transformers)

    # Test casting values
    vectorizer_cast = SuperVectorizer(
        cardinality_threshold=4,
        # we must have n_samples = 5 >= n_components
        high_card_cat_transformer=GapEncoder(n_components=2),
        numerical_transformer=StandardScaler(),
    )
    X_str = X.astype('object')
    # With pandas
    vectorizer_cast.fit_transform(X_str)
    check_same_transformers(expected_transformers_plain, vectorizer_cast.transformers)
    # With numpy
    vectorizer_cast.fit_transform(X_str.to_numpy())
    check_same_transformers(expected_transformers_np_cast, vectorizer_cast.transformers)


def test_with_clean_data():
    """
    Defines the expected returns of the vectorizer in different settings,
    and runs the tests with a clean dataset.
    """
    X = _get_clean_dataframe()
    # Define the transformers we'll use throughout the test.
    expected_transformers_df = {
        'numeric': ['int', 'float'],
        'low_card_cat': ['str1', 'cat1'],
        'high_card_cat': ['str2', 'cat2'],
    }
    expected_transformers_2 = {
        'low_card_cat': ['str1', 'str2', 'cat1', 'cat2'],
    }
    expected_transformers_np_no_cast = {
        'low_card_cat': [2, 4],
        'high_card_cat': [3, 5],
        'numeric': [0, 1]
    }
    expected_transformers_series = {
        'low_card_cat': ['cat1'],
    }
    expected_transformers_plain = {
        'high_card_cat': ['str2', 'cat2'],
        'low_card_cat': ['str1', 'cat1'],
        'numeric': ['int', 'float']
    }
    expected_transformers_np_cast = {
        'numeric': [0, 1],
        'low_card_cat': [2, 4],
        'high_card_cat': [3, 5],
    }
    _test_possibilities(
        X,
        expected_transformers_df,
        expected_transformers_2,
        expected_transformers_np_no_cast,
        expected_transformers_series,
        expected_transformers_plain,
        expected_transformers_np_cast,
    )


def test_with_dirty_data():
    """
    Defines the expected returns of the vectorizer in different settings,
    and runs the tests with a dataset containing missing values.
    """
    X = _get_dirty_dataframe()
    # Define the transformers we'll use throughout the test.
    expected_transformers_df = {
        'numeric': ['int', 'float'],
        'low_card_cat': ['str1', 'cat1'],
        'high_card_cat': ['str2', 'cat2'],
    }
    expected_transformers_2 = {
        'low_card_cat': ['str1', 'str2', 'cat1', 'cat2'],
    }
    expected_transformers_np_no_cast = {
        'low_card_cat': [2, 4],
        'high_card_cat': [3, 5],
        'numeric': [0, 1],
    }
    expected_transformers_series = {
        'low_card_cat': ['cat1'],
    }
    expected_transformers_plain = {
        'high_card_cat': ['str2', 'cat2'],
        'low_card_cat': ['str1', 'cat1'],
        'numeric': ['int', 'float']
    }
    expected_transformers_np_cast = {
        'numeric': [0, 1],
        'low_card_cat': [2, 4],
        'high_card_cat': [3, 5],
    }
    _test_possibilities(
        X,
        expected_transformers_df,
        expected_transformers_2,
        expected_transformers_np_no_cast,
        expected_transformers_series,
        expected_transformers_plain,
        expected_transformers_np_cast,
    )


def test_get_feature_names():
    X = _get_clean_dataframe()

    vectorizer_w_pass = SuperVectorizer(remainder='passthrough')
    vectorizer_w_pass.fit(X)

    if LooseVersion(sklearn.__version__) < LooseVersion('0.23'):
        with pytest.raises(NotImplementedError):
            # Prior to sklearn 0.23, ColumnTransformer.get_feature_names
            # with "passthrough" transformer(s) raises a NotImplementedError
            assert vectorizer_w_pass.get_feature_names()
            assert vectorizer_w_pass.get_feature_names_out()
    else:
        expected_feature_names_pass = [  # Order matters. If it doesn't, convert to set.
            'str1_private', 'str1_public',
            'str2_chef', 'str2_lawyer', 'str2_manager', 'str2_officer', 'str2_teacher',
            'cat1_no', 'cat1_yes', 'cat2_20K+', 'cat2_30K+', 'cat2_40K+', 'cat2_50K+', 'cat2_60K+',
            'int', 'float'
        ]
        assert vectorizer_w_pass.get_feature_names() == expected_feature_names_pass
        assert vectorizer_w_pass.get_feature_names_out() == expected_feature_names_pass

    vectorizer_w_drop = SuperVectorizer(remainder='drop')
    vectorizer_w_drop.fit(X)

    expected_feature_names_drop = [  # Order matters. If it doesn't, convert to set.
        'str1_private', 'str1_public',
        'str2_chef', 'str2_lawyer', 'str2_manager', 'str2_officer', 'str2_teacher',
        'cat1_no', 'cat1_yes', 'cat2_20K+', 'cat2_30K+', 'cat2_40K+', 'cat2_50K+', 'cat2_60K+'
    ]
    assert vectorizer_w_drop.get_feature_names() == expected_feature_names_drop
    assert vectorizer_w_drop.get_feature_names_out() == expected_feature_names_drop


def test_fit():
    # Simply checks sklearn's `check_is_fitted` function raises an error if
    # the SuperVectorizer is instantiated but not fitted.
    # See GH#193
    sup_vec = SuperVectorizer()
    with pytest.raises(NotFittedError):
        if LooseVersion(sklearn.__version__) >= LooseVersion('0.22'):
            assert check_is_fitted(sup_vec)
        else:
            assert check_is_fitted(sup_vec, attributes=dir(sup_vec))


def test_transform():
    X = _get_clean_dataframe()
    sup_vec = SuperVectorizer()
    sup_vec.fit(X)
    s = [34, 5.5, 'private', 'manager', 'yes', '60K+']
    x = np.array(s).reshape(1, -1)
    x_trans = sup_vec.transform(x)
    assert (x_trans == [[1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 34, 5.5]]).all()


def fit_transform_equiv():
    """
    We will test the equivalence between using `.fit_transform(X)`
    and `.fit(X).transform(X).`
    """
    X1 = _get_clean_dataframe()
    X2 = _get_dirty_dataframe()

    sup_vec1 = SuperVectorizer()
    sup_vec2 = SuperVectorizer()
    sup_vec3 = SuperVectorizer()
    sup_vec4 = SuperVectorizer()

    enc1_x1 = sup_vec1.fit_transform(X1)
    enc2_x1 = sup_vec2.fit(X1).transform(X1)

    enc1_x2 = sup_vec3.fit_transform(X2)
    enc2_x2 = sup_vec4.fit(X2).transform(X2)

    assert enc1_x1 == enc2_x1
    assert sup_vec1 == sup_vec2

    assert enc1_x2 == enc2_x2
    assert sup_vec3 == sup_vec4


if __name__ == '__main__':
    print('start test_super_vectorizer with clean df')
    test_with_clean_data()
    print('test_super_vectorizer with clean df passed')
    print('start test_super_vectorizer with dirty df')
    test_with_dirty_data()
    print('test_super_vectorizer with dirty df passed')
    print('start test_get_feature_names')
    test_get_feature_names()
    print('test_get_feature_names passed')
    print('start test_fit')
    test_fit()
    print('test_fit passed')
    print('start fit_transform_equiv')
    fit_transform_equiv()
    print('fit_transform_equiv passed')

    print('Done')
