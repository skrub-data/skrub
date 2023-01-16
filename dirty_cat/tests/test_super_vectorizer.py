from typing import Any, Tuple

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError

from dirty_cat import GapEncoder, SuperVectorizer
from dirty_cat._utils import parse_version


def check_same_transformers(expected_transformers: dict, actual_transformers: list):
    # Construct the dict from the actual transformers
    actual_transformers_dict = {name: cols for name, trans, cols in actual_transformers}
    assert actual_transformers_dict == expected_transformers


def type_equality(expected_type, actual_type):
    """
    Checks that the expected type is equal to the actual type,
    assuming object and str types are equivalent
    (considered as categorical by the SuperVectorizer).
    """
    if (isinstance(expected_type, object) or isinstance(expected_type, str)) and (
        isinstance(actual_type, object) or isinstance(actual_type, str)
    ):
        return True
    else:
        return expected_type == actual_type


def _get_clean_dataframe() -> pd.DataFrame:
    """
    Creates a simple DataFrame with various types of data,
    and without missing values.
    """
    return pd.DataFrame(
        {
            "int": pd.Series([15, 56, 63, 12, 44], dtype="int"),
            "float": pd.Series([5.2, 2.4, 6.2, 10.45, 9.0], dtype="float"),
            "str1": pd.Series(
                ["public", "private", "private", "private", "public"], dtype="string"
            ),
            "str2": pd.Series(
                ["officer", "manager", "lawyer", "chef", "teacher"], dtype="string"
            ),
            "cat1": pd.Series(["yes", "yes", "no", "yes", "no"], dtype="category"),
            "cat2": pd.Series(
                ["20K+", "40K+", "60K+", "30K+", "50K+"], dtype="category"
            ),
        }
    )


def _get_dirty_dataframe() -> pd.DataFrame:
    """
    Creates a simple DataFrame with some missing values.
    We'll use different types of missing values (np.nan, pd.NA, None)
    to test the robustness of the vectorizer.
    """
    return pd.DataFrame(
        {
            "int": pd.Series([15, 56, pd.NA, 12, 44], dtype="Int64"),
            "float": pd.Series([5.2, 2.4, 6.2, 10.45, np.nan], dtype="Float64"),
            "str1": pd.Series(
                ["public", np.nan, "private", "private", "public"], dtype="object"
            ),
            "str2": pd.Series(
                ["officer", "manager", None, "chef", "teacher"], dtype="object"
            ),
            "cat1": pd.Series([np.nan, "yes", "no", "yes", "no"], dtype="object"),
            "cat2": pd.Series(["20K+", "40K+", "60K+", "30K+", np.nan], dtype="object"),
        }
    )


def _get_numpy_array() -> np.ndarray:
    return np.array(
        [
            ["15", "56", pd.NA, "12", ""],
            ["?", "2.4", "6.2", "10.45", np.nan],
            ["public", np.nan, "private", "private", pd.NA],
            ["officer", "manager", None, "chef", "teacher"],
            [np.nan, "yes", "no", "yes", "no"],
            ["20K+", "40K+", "60K+", "30K+", np.nan],
        ]
    ).T


def _get_list_of_lists() -> list:
    return _get_numpy_array().tolist()


def _get_datetimes_dataframe() -> pd.DataFrame:
    """
    Creates a DataFrame with various date formats,
    already converted or to be converted.
    """
    return pd.DataFrame(
        {
            "pd_datetime": [
                pd.Timestamp("2019-01-01"),
                pd.Timestamp("2019-01-02"),
                pd.Timestamp("2019-01-03"),
                pd.Timestamp("2019-01-04"),
                pd.Timestamp("2019-01-05"),
            ],
            "np_datetime": [
                np.datetime64("2018-01-01"),
                np.datetime64("2018-01-02"),
                np.datetime64("2018-01-03"),
                np.datetime64("2018-01-04"),
                np.datetime64("2018-01-05"),
            ],
            "dmy-": [
                "11-12-2029",
                "02-12-2012",
                "11-09-2012",
                "13-02-2000",
                "10-11-2001",
            ],
            # "mdy-": ['11-13-2013',
            #          '02-12-2012',
            #          '11-31-2012',
            #          '05-02-2000',
            #          '10-11-2001'],
            "ymd/": [
                "2014/12/31",
                "2001/11/23",
                "2005/02/12",
                "1997/11/01",
                "2011/05/05",
            ],
            "ymd/_hms:": [
                "2014/12/31 00:31:01",
                "2014/12/30 00:31:12",
                "2014/12/31 23:31:23",
                "2015/12/31 01:31:34",
                "2014/01/31 00:32:45",
            ],
        }
    )


def _test_possibilities(X):
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
    expected_transformers_df = {
        "numeric": ["int", "float"],
        "low_card_cat": ["str1", "cat1"],
        "high_card_cat": ["str2", "cat2"],
    }
    vectorizer_base.fit_transform(X)
    check_same_transformers(expected_transformers_df, vectorizer_base.transformers)

    # Test with higher cardinality threshold and no numeric transformer
    expected_transformers_2 = {
        "low_card_cat": ["str1", "str2", "cat1", "cat2"],
    }
    vectorizer_default = SuperVectorizer()  # Using default values
    vectorizer_default.fit_transform(X)
    check_same_transformers(expected_transformers_2, vectorizer_default.transformers)

    # Test with a numpy array
    arr = X.to_numpy()
    # Instead of the columns names, we'll have the column indices.
    expected_transformers_np_no_cast = {
        "low_card_cat": [2, 4],
        "high_card_cat": [3, 5],
        "numeric": [0, 1],
    }
    vectorizer_base.fit_transform(arr)
    check_same_transformers(
        expected_transformers_np_no_cast, vectorizer_base.transformers
    )

    # Test with pandas series
    expected_transformers_series = {
        "low_card_cat": ["cat1"],
    }
    vectorizer_base.fit_transform(X["cat1"])
    check_same_transformers(expected_transformers_series, vectorizer_base.transformers)

    # Test casting values
    vectorizer_cast = SuperVectorizer(
        cardinality_threshold=4,
        # we must have n_samples = 5 >= n_components
        high_card_cat_transformer=GapEncoder(n_components=2),
        numerical_transformer=StandardScaler(),
    )
    X_str = X.astype("object")
    # With pandas
    expected_transformers_plain = {
        "high_card_cat": ["str2", "cat2"],
        "low_card_cat": ["str1", "cat1"],
        "numeric": ["int", "float"],
    }
    vectorizer_cast.fit_transform(X_str)
    check_same_transformers(expected_transformers_plain, vectorizer_cast.transformers)
    # With numpy
    expected_transformers_np_cast = {
        "numeric": [0, 1],
        "low_card_cat": [2, 4],
        "high_card_cat": [3, 5],
    }
    vectorizer_cast.fit_transform(X_str.to_numpy())
    check_same_transformers(expected_transformers_np_cast, vectorizer_cast.transformers)


def test_with_clean_data():
    """
    Defines the expected returns of the vectorizer in different settings,
    and runs the tests with a clean dataset.
    """
    _test_possibilities(_get_clean_dataframe())


def test_with_dirty_data() -> None:
    """
    Defines the expected returns of the vectorizer in different settings,
    and runs the tests with a dataset containing missing values.
    """
    _test_possibilities(_get_dirty_dataframe())


def test_auto_cast() -> None:
    """
    Tests that the SuperVectorizer automatic type detection works as expected.
    """
    vectorizer = SuperVectorizer()

    # Test datetime detection
    X = _get_datetimes_dataframe()

    expected_types_datetimes = {
        "pd_datetime": "datetime64[ns]",
        "np_datetime": "datetime64[ns]",
        "dmy-": "datetime64[ns]",
        "ymd/": "datetime64[ns]",
        "ymd/_hms:": "datetime64[ns]",
    }
    X_trans = vectorizer._auto_cast(X)
    for col in X_trans.columns:
        assert expected_types_datetimes[col] == X_trans[col].dtype

    # Test other types detection

    expected_types_clean_dataframe = {
        "int": "int64",
        "float": "float64",
        "str1": "object",
        "str2": "object",
        "cat1": "object",
        "cat2": "object",
    }

    X = _get_clean_dataframe()
    X_trans = vectorizer._auto_cast(X)
    for col in X_trans.columns:
        assert type_equality(expected_types_clean_dataframe[col], X_trans[col].dtype)

    # Test that missing values don't prevent type detection
    expected_types_dirty_dataframe = {
        "int": "float64",  # int type doesn't support nans
        "float": "float64",
        "str1": "object",
        "str2": "object",
        "cat1": "object",
        "cat2": "object",
    }

    X = _get_dirty_dataframe()
    X_trans = vectorizer._auto_cast(X)
    for col in X_trans.columns:
        assert type_equality(expected_types_dirty_dataframe[col], X_trans[col].dtype)


def test_with_arrays():
    """
    Check that the SuperVectorizer works if we input
    a list of lists or a numpy array.
    """
    expected_transformers = {
        "numeric": [0, 1],
        "low_card_cat": [2, 4],
        "high_card_cat": [3, 5],
    }
    vectorizer = SuperVectorizer(
        cardinality_threshold=4,
        # we must have n_samples = 5 >= n_components
        high_card_cat_transformer=GapEncoder(n_components=2),
        numerical_transformer=StandardScaler(),
    )

    X = _get_numpy_array()
    vectorizer.fit_transform(X)
    check_same_transformers(expected_transformers, vectorizer.transformers)

    X = _get_list_of_lists()
    vectorizer.fit_transform(X)
    check_same_transformers(expected_transformers, vectorizer.transformers)


def test_get_feature_names_out() -> None:
    X = _get_clean_dataframe()

    vec_w_pass = SuperVectorizer(remainder="passthrough")
    vec_w_pass.fit(X)

    # In this test, order matters. If it doesn't, convert to set.
    expected_feature_names_pass = [
        "str1_public",
        "str2_chef",
        "str2_lawyer",
        "str2_manager",
        "str2_officer",
        "str2_teacher",
        "cat1_yes",
        "cat2_20K+",
        "cat2_30K+",
        "cat2_40K+",
        "cat2_50K+",
        "cat2_60K+",
        "int",
        "float",
    ]
    if parse_version(sklearn.__version__) < parse_version("1.0"):
        assert vec_w_pass.get_feature_names() == expected_feature_names_pass
    else:
        assert vec_w_pass.get_feature_names_out() == expected_feature_names_pass

    vec_w_drop = SuperVectorizer(remainder="drop")
    vec_w_drop.fit(X)

    # In this test, order matters. If it doesn't, convert to set.
    expected_feature_names_drop = [
        "str1_public",
        "str2_chef",
        "str2_lawyer",
        "str2_manager",
        "str2_officer",
        "str2_teacher",
        "cat1_yes",
        "cat2_20K+",
        "cat2_30K+",
        "cat2_40K+",
        "cat2_50K+",
        "cat2_60K+",
    ]
    if parse_version(sklearn.__version__) < parse_version("1.0"):
        assert vec_w_drop.get_feature_names() == expected_feature_names_drop
    else:
        assert vec_w_drop.get_feature_names_out() == expected_feature_names_drop


def test_fit() -> None:
    # Simply checks sklearn's `check_is_fitted` function raises an error if
    # the SuperVectorizer is instantiated but not fitted.
    # See GH#193
    sup_vec = SuperVectorizer()
    with pytest.raises(NotFittedError):
        assert check_is_fitted(sup_vec)


def test_transform() -> None:
    X = _get_clean_dataframe()
    sup_vec = SuperVectorizer()
    sup_vec.fit(X)
    s = [34, 5.5, "private", "manager", "yes", "60K+"]
    x = np.array(s).reshape(1, -1)
    x_trans = sup_vec.transform(x)
    assert x_trans.tolist() == [
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 34.0, 5.5]
    ]
    # To understand the list above:
    # print(dict(zip(sup_vec.get_feature_names_out(), x_trans.tolist()[0])))


def test_fit_transform_equiv() -> None:
    """
    We will test the equivalence between using `.fit_transform(X)`
    and `.fit(X).transform(X).`
    """
    for X in [
        _get_clean_dataframe(),
        _get_dirty_dataframe(),
    ]:
        enc1_x1 = SuperVectorizer().fit_transform(X)
        enc2_x1 = SuperVectorizer().fit(X).transform(X)

        assert np.allclose(enc1_x1, enc2_x1, rtol=0, atol=0, equal_nan=True)


def _is_equal(elements: Tuple[Any, Any]) -> bool:
    """
    Fixture for values that return false when compared with `==`.
    """
    elem1, elem2 = elements  # Unpack
    return pd.isna(elem1) and pd.isna(elem2) or elem1 == elem2


def test_passthrough():
    """
    Tests that when passed no encoders, the SuperVectorizer
    returns the dataset as-is.
    """

    X_dirty = _get_dirty_dataframe()
    X_clean = _get_clean_dataframe()

    sv = SuperVectorizer(
        low_card_cat_transformer="passthrough",
        high_card_cat_transformer="passthrough",
        datetime_transformer="passthrough",
        numerical_transformer="passthrough",
        impute_missing="skip",
        auto_cast=False,
    )

    X_enc_dirty = pd.DataFrame(
        sv.fit_transform(X_dirty), columns=sv.get_feature_names_out()
    )
    X_enc_clean = pd.DataFrame(
        sv.fit_transform(X_clean), columns=sv.get_feature_names_out()
    )
    # Reorder encoded arrays' columns (see SV's doc "Notes" section as to why)
    X_enc_dirty = X_enc_dirty[X_dirty.columns]
    X_enc_clean = X_enc_clean[X_clean.columns]

    dirty_flat_df = X_dirty.to_numpy().ravel().tolist()
    dirty_flat_trans_df = X_enc_dirty.to_numpy().ravel().tolist()
    assert all(map(_is_equal, zip(dirty_flat_df, dirty_flat_trans_df)))
    assert (X_clean.to_numpy() == X_enc_clean.to_numpy()).all()

def test_check_fitted_super_vectorizer():
    """Test that calling transform before fit raises an error"""
    X = _get_clean_dataframe()
    sv = SuperVectorizer()
    with pytest.raises(NotFittedError):
        sv.transform(X)
    
    # Test that calling transform after fit works
    sv.fit(X)
    sv.transform(X)

