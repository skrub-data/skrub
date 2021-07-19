import pytest
import sklearn
import pandas as pd

from sklearn.preprocessing import StandardScaler

from distutils.version import LooseVersion

from dirty_cat import SuperVectorizer
from dirty_cat import GapEncoder


def check_same_transformers(expected_transformers: dict, actual_transformers: list):
    # Construct the dict from the actual transformers
    actual_transformers_dict = dict([(name, cols) for name, trans, cols in actual_transformers])
    assert actual_transformers_dict == expected_transformers


def _get_dataframe():
    return pd.DataFrame({
        'int': pd.Series([15, 56, 63, 12, 44], dtype=int),
        'float': pd.Series([5.2, 2.4, 6.2, 10.45, 9.], dtype=float),
        'str1': pd.Series(['public', 'private', 'private', 'private', 'public'], dtype='string'),
        'str2': pd.Series(['officer', 'manager', 'lawyer', 'chef', 'teacher'], dtype='string'),
        'cat1': pd.Series(['yes', 'yes', 'no', 'yes', 'no'], dtype='category'),
        'cat2': pd.Series(['20K+', '40K+', '60K+', '30K+', '50K+'], dtype='category'),
    })


def test_super_vectorizer():
    # Create a simple DataFrame
    X = _get_dataframe()
    # Test with low cardinality and a StandardScaler for the numeric columns
    vectorizer_base = SuperVectorizer(
        cardinality_threshold=3,
        # we must have n_samples = 5 >= n_components
        high_card_str_transformer=GapEncoder(n_components=2),
        high_card_cat_transformer=GapEncoder(n_components=2),
        numerical_transformer=StandardScaler(),
    )
    # Warning: order-dependant
    expected_transformers_df = {
        'numeric': ['int', 'float'],
        'low_card_str': ['str1'],
        'high_card_str': ['str2'],
        'low_card_cat': ['cat1'],
        'high_card_cat': ['cat2'],
    }
    vectorizer_base.fit_transform(X)
    check_same_transformers(expected_transformers_df, vectorizer_base.transformers)

    # Test with higher cardinality threshold and no numeric transformer
    vectorizer_default = SuperVectorizer()  # Using default values
    expected_transformers_2 = {
        'low_card_str': ['str1', 'str2'],
        'low_card_cat': ['cat1', 'cat2'],
    }
    vectorizer_default.fit_transform(X)
    check_same_transformers(expected_transformers_2, vectorizer_default.transformers)

    # Test with a numpy array
    arr = X.to_numpy()
    # Instead of the columns names, we'll have the column indices.
    expected_transformers_np = {
        'numeric': [0, 1],
        'low_card_str': [2, 4],
        'high_card_str': [3, 5],
    }
    vectorizer_base.fit_transform(arr)
    check_same_transformers(expected_transformers_np, vectorizer_base.transformers)

    # Test with pandas series
    expected_transformers_series = {
        'low_card_cat': ['cat1'],
    }
    vectorizer_base.fit_transform(X['cat1'])
    check_same_transformers(expected_transformers_series, vectorizer_base.transformers)

    # Test casting values
    vectorizer_cast = SuperVectorizer(
        cardinality_threshold=3,
        auto_cast=True,
        # we must have n_samples = 5 >= n_components
        high_card_str_transformer=GapEncoder(n_components=2),
        high_card_cat_transformer=GapEncoder(n_components=2),
        numerical_transformer=StandardScaler(),
    )
    X_str = X.astype('object')
    expected_transformers_plain = {
        'high_card_str': ['str2', 'cat2'],
        'low_card_str': ['str1', 'cat1'],
        'numeric': ['int', 'float']
    }
    # With pandas
    vectorizer_cast.fit_transform(X_str)
    check_same_transformers(expected_transformers_plain, vectorizer_cast.transformers)
    # With numpy
    vectorizer_cast.fit_transform(X_str.to_numpy())
    check_same_transformers(expected_transformers_np, vectorizer_cast.transformers)


def test_get_feature_names():
    X = _get_dataframe()

    vectorizer_w_pass = SuperVectorizer(remainder='passthrough')
    vectorizer_w_pass.fit(X)

    if LooseVersion(sklearn.__version__) < LooseVersion('0.23'):
        with pytest.raises(NotImplementedError):
            # Prior to sklearn 0.23, ColumnTransformer.get_feature_names
            # with "passthrough" transformer(s) raises a NotImplementedError
            assert vectorizer_w_pass.get_feature_names()
    else:
        expected_feature_names_pass = [  # Order matters. If it doesn't, convert to set.
            'str1_private', 'str1_public',
            'str2_chef', 'str2_lawyer', 'str2_manager', 'str2_officer', 'str2_teacher',
            'cat1_no', 'cat1_yes', 'cat2_20K+', 'cat2_30K+', 'cat2_40K+', 'cat2_50K+', 'cat2_60K+',
            'int', 'float'
        ]
        assert vectorizer_w_pass.get_feature_names() == expected_feature_names_pass

    vectorizer_w_drop = SuperVectorizer(remainder='drop')
    vectorizer_w_drop.fit(X)

    expected_feature_names_drop = [  # Order matters. If it doesn't, convert to set.
        'str1_private', 'str1_public',
        'str2_chef', 'str2_lawyer', 'str2_manager', 'str2_officer', 'str2_teacher',
        'cat1_no', 'cat1_yes', 'cat2_20K+', 'cat2_30K+', 'cat2_40K+', 'cat2_50K+', 'cat2_60K+'
    ]
    assert vectorizer_w_drop.get_feature_names() == expected_feature_names_drop


if __name__ == '__main__':
    print('start test_super_vectorizer')
    test_super_vectorizer()
    print('test_super_vectorizer passed')
    print('start test_get_feature_names')
    test_get_feature_names()
    print('test_get_feature_names passed')

    print('Done')
