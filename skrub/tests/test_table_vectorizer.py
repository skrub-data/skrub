import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils._testing import assert_array_equal, skip_if_no_parallel
from sklearn.utils.validation import check_is_fitted

from skrub import GapEncoder, MinHashEncoder, TableVectorizer
from skrub._datetime_encoder import _is_pandas_format_mixed_available
from skrub._table_vectorizer import _infer_date_format
from skrub.tests.utils import transformers_list_equal

MSG_PANDAS_DEPRECATED_WARNING = "Skip deprecation warning"


def check_same_transformers(
    expected_transformers: dict, actual_transformers: list
) -> None:
    # Construct the dict from the actual transformers
    actual_transformers_dict = {name: cols for name, trans, cols in actual_transformers}
    assert actual_transformers_dict == expected_transformers


def type_equality(expected_type, actual_type) -> bool:
    """
    Checks that the expected type is equal to the actual type,
    assuming object and str types are equivalent
    (considered as categorical by the TableVectorizer).
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


def _get_dirty_dataframe(categorical_dtype="object") -> pd.DataFrame:
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
                ["public", np.nan, "private", "private", "public"],
                dtype=categorical_dtype,
            ),
            "str2": pd.Series(
                ["officer", "manager", None, "chef", "teacher"],
                dtype=categorical_dtype,
            ),
            "cat1": pd.Series(
                [np.nan, "yes", "no", "yes", "no"], dtype=categorical_dtype
            ),
            "cat2": pd.Series(
                ["20K+", "40K+", "60K+", "30K+", np.nan], dtype=categorical_dtype
            ),
        }
    )


def _get_mixed_types_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "int_str": ["1", "2", 3, "3", 5],
            "float_str": ["1.0", pd.NA, 3.0, "3.0", 5.0],
            "int_float": [1, 2, 3.0, 3, 5.0],
            "bool_str": ["True", False, True, "False", "True"],
        }
    )


def _get_mixed_types_array() -> np.ndarray:
    return np.array(
        [
            ["1", "2", 3, "3", 5],
            ["1.0", np.nan, 3.0, "3.0", 5.0],
            [1, 2, 3.0, 3, 5.0],
            ["True", False, True, "False", "True"],
        ]
    ).T


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
            # this date format is not found by pandas guess_datetime_format
            # so shoulnd't be found by our _infer_datetime_format
            # but pandas.to_datetime can still parse it
            "mm/dd/yy": ["12/1/22", "2/3/05", "2/1/20", "10/7/99", "1/23/04"],
        }
    )


def _test_possibilities(X) -> None:
    """
    Do a bunch of tests with the TableVectorizer.
    We take some expected transformers results as argument. They're usually
    lists or dictionaries.
    """
    # Test with low cardinality and a StandardScaler for the numeric columns
    vectorizer_base = TableVectorizer(
        cardinality_threshold=4,
        # we must have n_samples = 5 >= n_components
        high_cardinality_transformer=GapEncoder(n_components=2),
        numerical_transformer=StandardScaler(),
    )
    # Warning: order-dependant
    expected_transformers_df = {
        "numeric": ["int", "float"],
        "low_card_cat": ["str1", "cat1"],
        "high_card_cat": ["str2", "cat2"],
    }
    vectorizer_base.fit_transform(X)
    check_same_transformers(expected_transformers_df, vectorizer_base.transformers_)

    # Test with higher cardinality threshold and no numeric transformer
    expected_transformers_2 = {
        "low_card_cat": ["str1", "str2", "cat1", "cat2"],
        "numeric": ["int", "float"],
    }
    vectorizer_default = TableVectorizer()  # Using default values
    vectorizer_default.fit_transform(X)
    check_same_transformers(expected_transformers_2, vectorizer_default.transformers_)

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
        expected_transformers_np_no_cast, vectorizer_base.transformers_
    )

    # Test with single column dataframe
    expected_transformers_series = {
        "low_card_cat": ["cat1"],
    }
    vectorizer_base.fit_transform(X[["cat1"]])
    check_same_transformers(expected_transformers_series, vectorizer_base.transformers_)

    # Test casting values
    vectorizer_cast = TableVectorizer(
        cardinality_threshold=4,
        # we must have n_samples = 5 >= n_components
        high_cardinality_transformer=GapEncoder(n_components=2),
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
    check_same_transformers(expected_transformers_plain, vectorizer_cast.transformers_)
    # With numpy
    expected_transformers_np_cast = {
        "numeric": [0, 1],
        "low_card_cat": [2, 4],
        "high_card_cat": [3, 5],
    }
    vectorizer_cast.fit_transform(X_str.to_numpy())
    check_same_transformers(
        expected_transformers_np_cast, vectorizer_cast.transformers_
    )


def test_duplicate_column_names() -> None:
    """
    Test to check if the tablevectorizer raises an error with
    duplicate column names
    """
    tablevectorizer = TableVectorizer()
    # Creates a simple dataframe with duplicate column names
    data = [(3, "a"), (2, "b"), (1, "c"), (0, "d")]
    X_dup_col_names = pd.DataFrame.from_records(data, columns=["col_1", "col_1"])

    with pytest.raises(AssertionError, match=r"Duplicate column names"):
        tablevectorizer.fit_transform(X_dup_col_names)


def test_with_clean_data() -> None:
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
    _test_possibilities(_get_dirty_dataframe(categorical_dtype="object"))
    _test_possibilities(_get_dirty_dataframe(categorical_dtype="category"))


def test_auto_cast() -> None:
    """
    Tests that the TableVectorizer automatic type detection works as expected.
    """
    vectorizer = TableVectorizer()

    # Test datetime detection
    X = _get_datetimes_dataframe()
    # Add weird index to test that it's not used
    X.index = [10, 3, 4, 2, 5]

    expected_types_datetimes = {
        "pd_datetime": "datetime64[ns]",
        "np_datetime": "datetime64[ns]",
        "dmy-": "datetime64[ns]",
        "ymd/": "datetime64[ns]",
        "ymd/_hms:": "datetime64[ns]",
        "mm/dd/yy": "datetime64[ns]",
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


def test_with_arrays() -> None:
    """
    Check that the TableVectorizer works if we input
    a list of lists or a numpy array.
    """
    expected_transformers = {
        "numeric": [0, 1],
        "low_card_cat": [2, 4],
        "high_card_cat": [3, 5],
    }
    vectorizer = TableVectorizer(
        cardinality_threshold=4,
        # we must have n_samples = 5 >= n_components
        high_cardinality_transformer=GapEncoder(n_components=2),
        numerical_transformer=StandardScaler(),
    )

    X = _get_numpy_array()
    vectorizer.fit_transform(X)
    check_same_transformers(expected_transformers, vectorizer.transformers_)

    X = _get_list_of_lists()
    vectorizer.fit_transform(X)
    check_same_transformers(expected_transformers, vectorizer.transformers_)


def test_get_feature_names_out() -> None:
    X = _get_clean_dataframe()

    vec_w_pass = TableVectorizer(remainder="passthrough")
    vec_w_pass.fit(X)

    # In this test, order matters. If it doesn't, convert to set.
    expected_feature_names_pass = np.array(
        [
            "int",
            "float",
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
    )
    assert_array_equal(
        vec_w_pass.get_feature_names_out(),
        expected_feature_names_pass,
    )

    vec_w_drop = TableVectorizer(remainder="drop")
    vec_w_drop.fit(X)

    # In this test, order matters. If it doesn't, convert to set.
    expected_feature_names_drop = [
        "int",
        "float",
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
    assert vec_w_drop.get_feature_names_out().tolist() == expected_feature_names_drop


def test_fit() -> None:
    # Simply checks sklearn's `check_is_fitted` function raises an error if
    # the TableVectorizer is instantiated but not fitted.
    # See GH#193
    table_vec = TableVectorizer()
    with pytest.raises(NotFittedError):
        assert check_is_fitted(table_vec)


def test_transform() -> None:
    X = _get_clean_dataframe()
    table_vec = TableVectorizer()
    table_vec.fit(X)
    s = [34, 5.5, "private", "manager", "yes", "60K+"]
    x = np.array(s).reshape(1, -1)
    x_trans = table_vec.transform(x)
    assert x_trans.tolist() == [
        [34.0, 5.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ]
    # To understand the list above:
    # print(dict(zip(table_vec.get_feature_names_out(), x_trans.tolist()[0])))


def test_fit_transform_equiv() -> None:
    """
    We will test the equivalence between using `.fit_transform(X)`
    and `.fit(X).transform(X).`
    """
    for X in [
        _get_clean_dataframe(),
        _get_dirty_dataframe(categorical_dtype="object"),
        _get_dirty_dataframe(categorical_dtype="category"),
        _get_mixed_types_dataframe(),
        _get_mixed_types_array(),
    ]:
        enc1_x1 = TableVectorizer().fit_transform(X)
        enc2_x1 = TableVectorizer().fit(X).transform(X)

        assert np.allclose(enc1_x1, enc2_x1, rtol=0, atol=0, equal_nan=True)


def _is_equal(elements: tuple[any, any]) -> bool:
    """
    Fixture for values that return false when compared with `==`.
    """
    elem1, elem2 = elements  # Unpack
    return pd.isna(elem1) and pd.isna(elem2) or elem1 == elem2


def test_passthrough() -> None:
    """
    Tests that when passed no encoders, the TableVectorizer
    returns the dataset as-is.
    """

    X_dirty = _get_dirty_dataframe()
    X_clean = _get_clean_dataframe()

    tv = TableVectorizer(
        low_cardinality_transformer="passthrough",
        high_cardinality_transformer="passthrough",
        datetime_transformer="passthrough",
        numerical_transformer="passthrough",
        impute_missing="skip",
        auto_cast=False,
    )

    X_enc_dirty = pd.DataFrame(
        tv.fit_transform(X_dirty), columns=tv.get_feature_names_out()
    )
    X_enc_clean = pd.DataFrame(
        tv.fit_transform(X_clean), columns=tv.get_feature_names_out()
    )
    # Reorder encoded arrays' columns
    # (see TableVectorizer's doc "Notes" section as to why)
    X_enc_dirty = X_enc_dirty[X_dirty.columns]
    X_enc_clean = X_enc_clean[X_clean.columns]

    dirty_flat_df = X_dirty.to_numpy().ravel().tolist()
    dirty_flat_trans_df = X_enc_dirty.to_numpy().ravel().tolist()
    assert all(map(_is_equal, zip(dirty_flat_df, dirty_flat_trans_df)))
    assert (X_clean.to_numpy() == X_enc_clean.to_numpy()).all()


def test_check_fitted_table_vectorizer() -> None:
    """Test that calling transform before fit raises an error"""
    X = _get_clean_dataframe()
    tv = TableVectorizer()
    with pytest.raises(NotFittedError):
        tv.transform(X)

    # Test that calling transform after fit works
    tv.fit(X)
    tv.transform(X)


def test_handle_unknown() -> None:
    """
    Test that new categories encountered in the test set
    are handled correctly.
    """
    X = _get_clean_dataframe()
    # Test with low cardinality and a StandardScaler for the numeric columns
    table_vec = TableVectorizer(
        cardinality_threshold=6,  # treat all columns as low cardinality
    )
    table_vec.fit(X)
    x_unknown = pd.DataFrame(
        {
            "int": pd.Series([3, 1], dtype="int"),
            "float": pd.Series([2.1, 4.3], dtype="float"),
            "str1": pd.Series(["semi-private", "public"], dtype="string"),
            "str2": pd.Series(["researcher", "chef"], dtype="string"),
            "cat1": pd.Series(["maybe", "yes"], dtype="category"),
            "cat2": pd.Series(["70K+", "20K+"], dtype="category"),
        }
    )
    x_known = pd.DataFrame(
        {
            "int": pd.Series([1, 4], dtype="int"),
            "float": pd.Series([4.3, 3.3], dtype="float"),
            "str1": pd.Series(["public", "private"], dtype="string"),
            "str2": pd.Series(["chef", "chef"], dtype="string"),
            "cat1": pd.Series(["yes", "no"], dtype="category"),
            "cat2": pd.Series(["30K+", "20K+"], dtype="category"),
        }
    )

    # Default behavior is "handle_unknown='ignore'",
    # so unknown categories are encoded as all zeros
    x_trans_unknown = table_vec.transform(x_unknown)
    x_trans_known = table_vec.transform(x_known)

    assert x_trans_unknown.shape == x_trans_known.shape
    n_zeroes = (
        X["str2"].nunique() + X["cat2"].nunique() + 2
    )  # 2 for binary columns which get one
    # cateogry dropped
    assert np.allclose(
        x_trans_unknown[0, 2:n_zeroes], np.zeros_like(x_trans_unknown[0, 2:n_zeroes])
    )
    assert x_trans_unknown[0, 0] != 0
    assert not np.allclose(
        x_trans_known[0, :n_zeroes], np.zeros_like(x_trans_known[0, :n_zeroes])
    )


def test__infer_date_format() -> None:
    # Test with an ambiguous date format
    # but with a single format that works for all rows
    date_column = pd.Series(["01-01-2022", "13-01-2022", "01-03-2022"])
    assert _infer_date_format(date_column) == "%d-%m-%Y"

    date_column = pd.Series(["01-01-2022", "01-13-2022", "01-03-2022"])
    assert _infer_date_format(date_column) == "%m-%d-%Y"

    # Test with an ambiguous date format
    # but several formats that work for all rows
    date_column = pd.Series(["01-01-2022", "01-02-2019", "01-03-2019"])
    # check that a warning is raised
    with pytest.warns(UserWarning):
        assert _infer_date_format(date_column) == "%m-%d-%Y"

    # Test with irreconcilable date formats
    date_column = pd.Series(["01-01-2022", "13-01-2019", "01-03-2022", "01-13-2019"])
    assert _infer_date_format(date_column) is None

    # Test previous cases with missing values

    date_column = pd.Series(["01-01-2022", "13-01-2022", "01-03-2022", pd.NA])
    assert _infer_date_format(date_column) == "%d-%m-%Y"

    date_column = pd.Series(["01-01-2022", "01-13-2022", "01-03-2022", pd.NA])
    assert _infer_date_format(date_column) == "%m-%d-%Y"

    date_column = pd.Series(["01-01-2022", "01-02-2019", "01-03-2019", pd.NA])
    # check that a warning is raised
    with pytest.warns(UserWarning):
        assert _infer_date_format(date_column) == "%m-%d-%Y"

    date_column = pd.Series(
        ["01-01-2022", "13-01-2019", "01-03-2022", "01-13-2019", pd.NA]
    )
    assert _infer_date_format(date_column) is None

    # Test previous cases with hours and minutes

    date_column = pd.Series(
        ["01-01-2022 12:00", "13-01-2022 12:00", "01-03-2022 12:00"]
    )
    assert _infer_date_format(date_column) == "%d-%m-%Y %H:%M"

    date_column = pd.Series(
        ["01-01-2022 12:00", "01-13-2022 12:00", "01-03-2022 12:00"]
    )
    assert _infer_date_format(date_column) == "%m-%d-%Y %H:%M"

    date_column = pd.Series(
        ["01-01-2022 12:00", "01-02-2019 12:00", "01-03-2019 12:00"]
    )
    # check that a warning is raised
    with pytest.warns(UserWarning):
        assert _infer_date_format(date_column) == "%m-%d-%Y %H:%M"

    date_column = pd.Series(
        [
            "01-01-2022 12:00",
            "13-01-2019 12:00",
            "01-03-2022 12:00",
            "01-13-2019 12:00",
        ]
    )
    assert _infer_date_format(date_column) is None

    # Test with an empty column
    date_column = pd.Series([], dtype="object")
    assert _infer_date_format(date_column) is None

    # Test with a column containing only NaN values
    date_column = pd.Series([pd.NA, pd.NA, pd.NA])
    assert _infer_date_format(date_column) is None

    # Test with a column containing both dates and non-dates
    date_column = pd.Series(["2022-01-01", "2022-01-02", "not a date"])
    assert _infer_date_format(date_column) is None

    # Test with a column containing more than two date formats
    date_column = pd.Series(["2022-01-01", "01/02/2022", "20220103", "2022-Jan-04"])
    assert _infer_date_format(date_column) is None


@pytest.mark.parametrize(
    ["specific_transformers", "expected_transformers_"],
    [
        (
            (MinHashEncoder(), ["str1", "str2"]),
            [
                ("numeric", "passthrough", ["int", "float"]),
                ("minhashencoder", "MinHashEncoder", ["str1", "str2"]),
                ("low_card_cat", "OneHotEncoder", ["cat1", "cat2"]),
            ],
        ),
        (
            ("mh_cat1", MinHashEncoder(), ["cat1"]),
            [
                ("numeric", "passthrough", ["int", "float"]),
                ("mh_cat1", "MinHashEncoder", ["cat1"]),
                ("low_card_cat", "OneHotEncoder", ["str1", "str2", "cat2"]),
            ],
        ),
    ],
)
def test_specifying_specific_column_transformer(
    specific_transformers, expected_transformers_
) -> None:
    """
    Test that specifying a specific transformer works as expected.
    """
    X = _get_dirty_dataframe()
    tv = TableVectorizer(specific_transformers=[specific_transformers]).fit(X)
    clean_transformers_ = [
        (
            (name, transformer.__class__.__name__, columns)
            if not isinstance(transformer, str)
            else (name, transformer, columns)
        )
        for name, transformer, columns in tv.transformers_
    ]
    # Sort to ignore order
    assert sorted(clean_transformers_) == sorted(expected_transformers_)


def test_specific_transformers_unexpected_behavior():
    """
    Tests a few cases of unsupported behavior when using `specific_transformers`.
    """
    X = _get_clean_dataframe()

    # Test that using tuple lengths other than 2 or 3 raises an error
    with pytest.raises(TypeError):
        assert TableVectorizer(specific_transformers=[(StandardScaler(),)]).fit(X)
        assert TableVectorizer(
            specific_transformers=[("dummy", StandardScaler(), ["float"], 1)]
        ).fit(X)

    # Test that using mixed length tuples raises an error
    with pytest.raises(TypeError):
        assert TableVectorizer(
            specific_transformers=[
                (StandardScaler(), ["float"]),
                ("dummy", StandardScaler(), ["float"]),
            ]
        ).fit(X)


@pytest.mark.parametrize(
    "pipeline",
    [
        TableVectorizer(),
        TableVectorizer(
            specific_transformers=[
                (MinHashEncoder(), ["cat1", "cat2"]),
            ],
        ),
        TableVectorizer(
            low_cardinality_transformer=MinHashEncoder(),
        ),
    ],
)
def test_deterministic(pipeline) -> None:
    """
    Tests that running the same TableVectorizer multiple times with the same
    (deterministic) components results in the same output.
    """
    X = _get_dirty_dataframe()
    X_enc_prev = pipeline.fit_transform(X)
    for _ in range(5):
        X_enc = pipeline.fit_transform(X)
        np.testing.assert_array_equal(X_enc, X_enc_prev)
        X_enc_prev = X_enc


def test_mixed_types() -> None:
    # TODO: datetime/str mixed types
    # don't work
    df = _get_mixed_types_dataframe()
    table_vec = TableVectorizer()
    table_vec.fit_transform(df)
    # check that the types are correctly inferred
    table_vec.fit_transform(df)
    expected_transformers_df = {
        "numeric": ["int_str", "float_str", "int_float"],
        "low_card_cat": ["bool_str"],
    }
    check_same_transformers(expected_transformers_df, table_vec.transformers_)

    X = _get_mixed_types_array()
    table_vec = TableVectorizer()
    table_vec.fit_transform(X)
    # check that the types are correctly inferred
    table_vec.fit_transform(X)
    expected_transformers_array = {
        "numeric": [0, 1, 2],
        "low_card_cat": [3],
    }
    check_same_transformers(expected_transformers_array, table_vec.transformers_)


@pytest.mark.parametrize(
    "X_fit, X_transform_original, X_transform_with_missing_original",
    [
        # All nans during fit, 1 category during transform
        (
            pd.DataFrame({"col1": [np.nan, np.nan, np.nan]}),
            pd.DataFrame({"col1": [np.nan, np.nan, "placeholder"]}),
            pd.DataFrame({"col1": [np.nan, np.nan, np.nan]}),
        ),
        # All floats during fit, 1 category during transform
        (
            pd.DataFrame({"col1": [1.0, 2.0, 3.0]}),
            pd.DataFrame({"col1": [1.0, 2.0, "placeholder"]}),
            pd.DataFrame({"col1": [1.0, 2.0, np.nan]}),
        ),
        # All datetimes during fit, 1 category during transform
        pytest.param(
            pd.DataFrame(
                {
                    "col1": [
                        pd.Timestamp("2019-01-01"),
                        pd.Timestamp("2019-01-02"),
                        pd.Timestamp("2019-01-03"),
                    ]
                }
            ),
            pd.DataFrame(
                {
                    "col1": [
                        pd.Timestamp("2019-01-01"),
                        pd.Timestamp("2019-01-02"),
                        "placeholder",
                    ]
                }
            ),
            pd.DataFrame(
                {
                    "col1": [
                        pd.Timestamp("2019-01-01"),
                        pd.Timestamp("2019-01-02"),
                        np.nan,
                    ]
                }
            ),
            marks=pytest.mark.skipif(
                not _is_pandas_format_mixed_available(),
                reason=MSG_PANDAS_DEPRECATED_WARNING,
            ),
        ),
    ],
)
def test_changing_types(
    X_fit, X_transform_original, X_transform_with_missing_original
) -> None:
    """
    Test that the TableVectorizer performs properly when the
    type inferred during fit does not match the type of the
    data during transform.
    """
    for new_category in ["a", "new category", "[test]"]:
        table_vec = TableVectorizer()
        table_vec.fit_transform(X_fit)
        expected_dtype = table_vec.types_[0]
        # convert [ and ] to \\[ and \\] to avoid pytest warning
        expected_dtype = str(expected_dtype).replace("[", "\\[").replace("]", "\\]")
        new_category_regex = str(new_category).replace("[", "\\[").replace("]", "\\]")
        expected_warning_msg = (
            f".*'{new_category_regex}'.*could not be converted.*{expected_dtype}.*"
        )

        # replace "placeholder" with the new category
        X_transform = X_transform_original.replace("placeholder", new_category)
        X_transform_with_missing = X_transform_with_missing_original.replace(
            "placeholder", new_category
        )
        with pytest.warns(UserWarning, match=expected_warning_msg):
            res = table_vec.transform(X_transform)
        # the TableVectorizer should behave as if the new entry
        # with the wrong type was missing
        res_missing = table_vec.transform(X_transform_with_missing)
        assert np.allclose(res, res_missing, equal_nan=True)


def test_changing_types_int_float() -> None:
    # The TableVectorizer shouldn't cast floats to ints
    # even if only ints were seen during fit
    X_fit, X_transform = (
        pd.DataFrame(pd.Series([1, 2, 3])),
        pd.DataFrame(pd.Series([1, 2, 3.3])),
    )
    table_vec = TableVectorizer()
    table_vec.fit_transform(X_fit)
    res = table_vec.transform(X_transform)
    assert np.allclose(res, np.array([[1.0], [2.0], [3.3]]))


def test_column_by_column() -> None:
    # Test that the TableVectorizer gives the same result
    # when applied column by column
    X = _get_clean_dataframe()
    table_vec_all_cols = TableVectorizer(
        high_cardinality_transformer=GapEncoder(n_components=2, random_state=0),
        cardinality_threshold=4,
    )
    table_vec_all_cols.fit(X)
    for col in X.columns:
        table_vec_one_col = TableVectorizer(
            high_cardinality_transformer=GapEncoder(n_components=2, random_state=0),
            cardinality_threshold=4,
        )
        table_vec_one_col.fit(X[[col]])
        features_from_col = [
            feat
            for feat in table_vec_all_cols.get_feature_names_out()
            if feat.startswith(col)
        ]
        assert table_vec_one_col.get_feature_names_out().tolist() == features_from_col
        indices_features_from_col = [
            table_vec_all_cols.get_feature_names_out().tolist().index(feat)
            for feat in features_from_col
        ]
        excepted_result = table_vec_all_cols.transform(X)[:, indices_features_from_col]
        assert np.allclose(
            table_vec_one_col.transform(X[[col]]),
            excepted_result,
        )


@skip_if_no_parallel
@pytest.mark.parametrize(
    "high_cardinality_transformer",
    # the gap encoder and the minhashencoder
    # should be parallelized on all columns
    # the one hot encoder should not be parallelized
    [
        GapEncoder(n_components=2, random_state=0),
        OneHotEncoder(),
        MinHashEncoder(n_components=2),
    ],
)
def test_parallelism(high_cardinality_transformer) -> None:
    # Test that parallelism works
    X = _get_clean_dataframe()
    table_vec_no_parallel = TableVectorizer(
        high_cardinality_transformer=high_cardinality_transformer,
        cardinality_threshold=4,
    )
    X_trans = table_vec_no_parallel.fit_transform(X)
    with joblib.parallel_backend("loky"):
        for n_jobs in [None, 2, -1]:
            table_vec = TableVectorizer(
                n_jobs=n_jobs,
                high_cardinality_transformer=high_cardinality_transformer,
                cardinality_threshold=4,
            )
            X_trans_parallel = table_vec.fit_transform(X)
            assert_array_equal(X_trans, X_trans_parallel)
            assert table_vec.n_jobs == n_jobs
            # assert that all attributes are equal except for
            # the n_jobs attribute
            assert transformers_list_equal(
                table_vec.transformers_,
                table_vec_no_parallel.transformers_,
                ignore_params=("n_jobs",),
            )
            assert (
                table_vec.feature_names_in_ == table_vec_no_parallel.feature_names_in_
            ).all()
            assert table_vec.types_ == table_vec_no_parallel.types_
            assert table_vec.imputed_columns_ == table_vec_no_parallel.imputed_columns_
            # assert that get_feature_names_out gives the same result
            assert_array_equal(
                table_vec.get_feature_names_out(),
                table_vec_no_parallel.get_feature_names_out(),
            )
            # assert that get_params gives the same result expect for n_jobs
            # remove n_jobs from the dict
            params = table_vec.get_params()
            params.pop("n_jobs")
            params_no_parallel = table_vec_no_parallel.get_params()
            params_no_parallel.pop("n_jobs")
            assert str(params) == str(params_no_parallel)
            # assert that transform gives the same result
            assert_array_equal(
                table_vec.transform(X), table_vec_no_parallel.transform(X)
            )


def test_table_vectorizer_policy_propagate_n_jobs():
    """Check the propagation policy of `n_jobs` to the underlying transformers.

    We need to check that when `TableVectorizer.n_jobs` is set, then all underlying
    transformers `n_jobs` will be set to this value, except if the user provide a
    transformer in the constructor with the value `n_jobs` already set.
    """
    X = _get_clean_dataframe()

    # 1. Case where `TableVectorizer.n_jobs` is `None` and we should not propagate
    class DummyTransformerWithJobs(FunctionTransformer):
        def __init__(self, n_jobs=None):
            super().__init__()
            self.n_jobs = n_jobs

    table_vectorizer = TableVectorizer(
        numerical_transformer=DummyTransformerWithJobs(n_jobs=None),
        low_cardinality_transformer=DummyTransformerWithJobs(n_jobs=None),
        n_jobs=None,
    ).fit(X)
    assert table_vectorizer.named_transformers_["numeric"].n_jobs is None
    assert table_vectorizer.named_transformers_["low_card_cat"].n_jobs is None

    table_vectorizer = TableVectorizer(
        numerical_transformer=DummyTransformerWithJobs(n_jobs=2),
        low_cardinality_transformer=DummyTransformerWithJobs(n_jobs=None),
        n_jobs=None,
    ).fit(X)
    assert table_vectorizer.named_transformers_["numeric"].n_jobs == 2
    assert table_vectorizer.named_transformers_["low_card_cat"].n_jobs is None

    # 2. Case where `TableVectorizer.n_jobs` is not `None` and we should propagate
    # when the underlying transformer `n_jobs` is not set explicitly.
    table_vectorizer = TableVectorizer(
        numerical_transformer=DummyTransformerWithJobs(n_jobs=None),
        low_cardinality_transformer=DummyTransformerWithJobs(n_jobs=None),
        n_jobs=2,
    ).fit(X)
    assert table_vectorizer.named_transformers_["numeric"].n_jobs == 2
    assert table_vectorizer.named_transformers_["low_card_cat"].n_jobs == 2

    # 3. Case where `TableVectorizer.n_jobs` is not `None` and we should not propagate
    # when the underlying transformer `n_jobs` is set explicitly.
    table_vectorizer = TableVectorizer(
        numerical_transformer=DummyTransformerWithJobs(n_jobs=4),
        low_cardinality_transformer=DummyTransformerWithJobs(n_jobs=None),
        n_jobs=2,
    ).fit(X)
    assert table_vectorizer.named_transformers_["numeric"].n_jobs == 4
    assert table_vectorizer.named_transformers_["low_card_cat"].n_jobs == 2


def test_table_vectorizer_remainder_cloning():
    """Check that remainder is cloned when used."""
    df1 = _get_clean_dataframe()
    df2 = _get_datetimes_dataframe()
    df = pd.concat([df1, df2], axis=1)
    remainder = FunctionTransformer()
    table_vectorizer = TableVectorizer(
        low_cardinality_transformer="remainder",
        high_cardinality_transformer="remainder",
        numerical_transformer="remainder",
        datetime_transformer="remainder",
        remainder=remainder,
    ).fit(df)
    assert table_vectorizer.low_cardinality_transformer_ is not remainder
    assert table_vectorizer.high_cardinality_transformer_ is not remainder
    assert table_vectorizer.numerical_transformer_ is not remainder
    assert table_vectorizer.datetime_transformer_ is not remainder


def test_pandas_sparse_array():
    df = pd.DataFrame(
        dict(
            a=[1, 2, 3, 4, 5],
            b=[1, 0, 0, 0, 2],
        )
    )
    df["b"] = pd.arrays.SparseArray(df["b"])

    match = r"(?=.*sparse Pandas series)(?=.*'b')"
    with pytest.raises(TypeError, match=match):
        TableVectorizer().fit(df)

    df = df.astype(pd.SparseDtype())

    match = r"(?=.*sparse Pandas series)(?=.*'a', 'b')"
    with pytest.raises(TypeError, match=match):
        TableVectorizer().fit(df)
