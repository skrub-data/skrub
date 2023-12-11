import joblib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_raises
from pandas.testing import assert_frame_equal
from scipy.sparse import csr_matrix
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils._testing import skip_if_no_parallel

from skrub._datetime_encoder import DatetimeEncoder, _is_pandas_format_mixed_available
from skrub._gap_encoder import GapEncoder
from skrub._minhash_encoder import MinHashEncoder
from skrub._table_vectorizer import LOW_CARDINALITY_TRANSFORMER, TableVectorizer
from skrub.tests.utils import transformers_list_equal

MSG_PANDAS_DEPRECATED_WARNING = "Skip deprecation warning"


def type_equality(expected_type, actual_type):
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


def _get_clean_dataframe():
    """
    Creates a simple DataFrame with various types of data,
    and without missing values.
    """
    return pd.DataFrame(
        {
            "int": pd.Series([15, 56, 63, 12, 44], dtype="int64"),
            "float": pd.Series([5.2, 2.4, 6.2, 10.45, 9.0], dtype="float64"),
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


def _get_dirty_dataframe(categorical_dtype="object"):
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


def _get_mixed_types_dataframe():
    return pd.DataFrame(
        {
            "int_str": ["1", "2", 3, "3", 5],
            "float_str": ["1.0", pd.NA, 3.0, "3.0", 5.0],
            "int_float": [1, 2, 3.0, 3, 5.0],
            "bool_str": ["True", False, True, "False", "True"],
        }
    )


def _get_mixed_types_array():
    return np.array(
        [
            ["1", "2", 3, "3", 5],
            ["1.0", np.nan, 3.0, "3.0", 5.0],
            [1, 2, 3.0, 3, 5.0],
            ["True", False, True, "False", "True"],
        ]
    ).T


def _get_datetimes_dataframe():
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


def test_fit_default_transform():
    X = _get_clean_dataframe()
    vectorizer = TableVectorizer()
    vectorizer.fit(X)

    low_cardinality_cols = ["str1", "str2", "cat1", "cat2"]
    low_cardinality_transformer = LOW_CARDINALITY_TRANSFORMER.fit(
        X[low_cardinality_cols]
    )
    expected_transformers = [
        ("numeric", "passthrough", ["int", "float"]),
        ("low_cardinality", low_cardinality_transformer, low_cardinality_cols),
    ]

    assert transformers_list_equal(
        expected_transformers,
        vectorizer.transformers_,
        ignore_params=["categories_"],  # list of array of different lengths
    )
    categories = vectorizer.transformers_[1][1].categories_
    expected_categories = low_cardinality_transformer.categories_
    for categories_, expected_categories_ in zip(categories, expected_categories):
        assert_array_equal(categories_, expected_categories_)


def test_duplicate_column_names():
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


X = _get_datetimes_dataframe()
# Add weird index to test that it's not used
X.index = [10, 3, 4, 2, 5]

X_tuples = [
    (
        X,
        {
            "pd_datetime": "datetime64[ns]",
            "np_datetime": "datetime64[ns]",
            "dmy-": "datetime64[ns]",
            "ymd/": "datetime64[ns]",
            "ymd/_hms:": "datetime64[ns]",
            "mm/dd/yy": "datetime64[ns]",
        },
    ),
    # Test other types detection
    (
        _get_clean_dataframe(),
        {
            "int": "int64",
            "float": "float64",
            "str1": "object",
            "str2": "object",
            "cat1": "category",
            "cat2": "category",
        },
    ),
    (
        _get_dirty_dataframe("category"),
        {
            "int": "float64",  # int type doesn't support nans
            "float": "float64",
            "str1": "category",
            "str2": "category",
            "cat1": "category",
            "cat2": "category",
        },
    ),
]


@pytest.mark.parametrize("X, dict_expected_types", X_tuples)
def test_auto_cast(X, dict_expected_types):
    """
    Tests that the TableVectorizer automatic type detection works as expected.
    """
    vectorizer = TableVectorizer()
    X_trans = vectorizer._auto_cast(X, reset=True)
    for col in X_trans.columns:
        assert dict_expected_types[col] == X_trans[col].dtype


def test_auto_cast_missing_categories():
    X = _get_dirty_dataframe("category")
    vectorizer = TableVectorizer()
    _ = vectorizer._auto_cast(X, reset=True)

    expected_type_per_column = {
        "int": np.dtype("float64"),
        "float": np.dtype("float64"),
        "str1": pd.CategoricalDtype(
            categories=["private", "public"],
        ),
        "str2": pd.CategoricalDtype(
            categories=["chef", "manager", "officer", "teacher"],
        ),
        "cat1": pd.CategoricalDtype(
            categories=["no", "yes"],
        ),
        "cat2": pd.CategoricalDtype(
            categories=["20K+", "30K+", "40K+", "60K+"],
        ),
    }
    assert vectorizer.inferred_column_types_ == expected_type_per_column

    X = _get_dirty_dataframe("category")
    X_train = X.head(3).reset_index(drop=True)
    X_test = X.tail(2).reset_index(drop=True)
    _ = vectorizer._auto_cast(X_train, reset=True)
    _ = vectorizer._auto_cast(X_test, reset=False)

    assert vectorizer.inferred_column_types_ == expected_type_per_column


assert_tuples = [
    (
        dict(remainder="passthrough"),
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
        ],
    ),
    (
        dict(remainder="drop"),
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
        ],
    ),
]


@pytest.mark.parametrize("params, expected_features", assert_tuples)
def test_get_feature_names_out(params, expected_features):
    X = _get_clean_dataframe()
    vectorizer = TableVectorizer(**params).fit(X)
    assert_array_equal(
        vectorizer.get_feature_names_out(),
        expected_features,
    )


def test_transform():
    X = _get_clean_dataframe()
    table_vec = TableVectorizer().fit(X)
    x = pd.DataFrame(
        [
            {
                "int": 34,
                "float": 5.5,
                "str1": "private",
                "str2": "manager",
                "cat1": "yes",
                "cat2": "60K+",
            }
        ]
    )
    x_trans = table_vec.transform(x)
    expected_x_trans = [
        [34.0, 5.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ]
    assert_array_equal(x_trans, expected_x_trans)


inputs = [
    _get_clean_dataframe(),
    _get_dirty_dataframe(categorical_dtype="object"),
    _get_dirty_dataframe(categorical_dtype="category"),
    _get_mixed_types_dataframe(),
    _get_mixed_types_array(),
]


@pytest.mark.parametrize("X", inputs)
def test_fit_transform_equiv(X):
    # TODO: this check is probably already performed in test_sklearn
    """
    We will test the equivalence between using `.fit_transform(X)`
    and `.fit(X).transform(X).`
    """
    X_trans_1 = TableVectorizer().fit_transform(X)
    X_trans_2 = TableVectorizer().fit(X).transform(X)
    assert_array_equal(X_trans_1, X_trans_2)


inputs = [
    _get_dirty_dataframe(),
    _get_clean_dataframe(),
]


@pytest.mark.parametrize("X", inputs)
def test_passthrough(X):
    """
    Tests that when passed no encoders, the TableVectorizer
    returns the dataset as-is.
    """
    vectorizer = TableVectorizer(
        low_cardinality_transformer="passthrough",
        high_cardinality_transformer="passthrough",
        datetime_transformer="passthrough",
        numerical_transformer="passthrough",
        auto_cast=False,
    )
    vectorizer.set_output(transform="pandas")
    X_trans = vectorizer.fit_transform(X)

    assert_frame_equal(X.astype("object"), X_trans)


def test_handle_unknown_category():
    X = _get_clean_dataframe()
    # Treat all columns as low cardinality
    table_vec = TableVectorizer(cardinality_threshold=6).fit(X)
    X_unknown = pd.DataFrame(
        {
            "int": pd.Series([3, 1], dtype="int"),
            "float": pd.Series([2.1, 4.3], dtype="float"),
            "str1": pd.Series(["semi-private", "public"], dtype="string"),
            "str2": pd.Series(["researcher", "chef"], dtype="string"),
            "cat1": pd.Series(["maybe", "yes"], dtype="category"),
            "cat2": pd.Series(["70K+", "20K+"], dtype="category"),
        }
    )
    X_known = pd.DataFrame(
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
    X_trans_unknown = table_vec.transform(X_unknown)
    X_trans_known = table_vec.transform(X_known)

    assert X_trans_unknown.shape == X_trans_known.shape

    # +2 for binary columns which get one category dropped
    n_zeroes = X["str2"].nunique() + X["cat2"].nunique() + 2
    assert_array_equal(
        X_trans_unknown[0, 2:n_zeroes], np.zeros_like(X_trans_unknown[0, 2:n_zeroes])
    )
    assert_raises(
        AssertionError,
        assert_array_equal,
        X_trans_known[0, :n_zeroes],
        np.zeros_like(X_trans_known[0, :n_zeroes]),
    )


@pytest.mark.parametrize(
    ["specific_transformers", "expected_transformers_"],
    [
        (
            (MinHashEncoder(), ["str1", "str2"]),
            [
                ("numeric", "passthrough", ["int", "float"]),
                ("minhashencoder", "MinHashEncoder", ["str1", "str2"]),
                ("low_cardinality", "OneHotEncoder", ["cat1", "cat2"]),
            ],
        ),
        (
            ("mh_cat1", MinHashEncoder(), ["cat1"]),
            [
                ("numeric", "passthrough", ["int", "float"]),
                ("mh_cat1", "MinHashEncoder", ["cat1"]),
                ("low_cardinality", "OneHotEncoder", ["str1", "str2", "cat2"]),
            ],
        ),
    ],
)
def test_specifying_specific_column_transformer(
    specific_transformers, expected_transformers_
):
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


error_tuples = [
    (
        [(StandardScaler(),)],
        r"(?=.*got a list of tuples of length 1)",
    ),
    (
        [("dummy", StandardScaler(), ["float"], 1)],
        r"(?=.*got a list of tuples of length 4)",
    ),
    (
        [
            (StandardScaler(), ["float"]),
            ("dummy", StandardScaler(), ["float"]),
        ],
        r"(?=.*got length 3 at index 1)",
    ),
]


@pytest.mark.parametrize("specific_transformers, error_msg", error_tuples)
def test_specific_transformers_unexpected_behavior(specific_transformers, error_msg):
    """
    Test that using tuple lengths other than 2 or 3 raises an error
    """
    X = _get_clean_dataframe()

    with pytest.raises(TypeError, match=error_msg):
        TableVectorizer(specific_transformers=specific_transformers).fit(X)


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
def test_deterministic(pipeline):
    """
    Tests that running the same TableVectorizer multiple times with the same
    (deterministic) components results in the same output.
    """
    X = _get_dirty_dataframe()
    X_trans_1 = pipeline.fit_transform(X)
    X_trans_2 = pipeline.fit_transform(X)
    assert_array_equal(X_trans_1, X_trans_2)


def test_mixed_types():
    """
    Check that the types are correctly inferred.
    """
    X = _get_mixed_types_dataframe()
    vectorizer = TableVectorizer()
    vectorizer.fit(X)
    expected_name_to_columns = {
        "numeric": ["int_str", "float_str", "int_float"],
        "low_cardinality": ["bool_str"],
    }
    name_to_columns = {name: columns for name, _, columns in vectorizer.transformers_}
    assert expected_name_to_columns == name_to_columns


@pytest.mark.parametrize(
    "X_train, X_test, expected_X_out",
    [
        # All nans during fit, 1 category during transform
        (
            pd.DataFrame({"col1": [np.nan, np.nan, np.nan]}),
            pd.DataFrame({"col1": [np.nan, np.nan, "placeholder"]}),
            np.array([[np.nan], [np.nan], [np.nan]], dtype="float64"),
        ),
        # All floats during fit, 1 category during transform
        (
            pd.DataFrame({"col1": [1.0, 2.0, 3.0]}),
            pd.DataFrame({"col1": [1.0, 2.0, "placeholder"]}),
            np.array([[1.0], [2.0], [np.nan]]),
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
            np.array([[1.5463008e09], [1.5463872e09], [np.nan]]),
            marks=pytest.mark.skipif(
                not _is_pandas_format_mixed_available(),
                reason=MSG_PANDAS_DEPRECATED_WARNING,
            ),
        ),
    ],
)
def test_changing_types(X_train, X_test, expected_X_out):
    """
    Test that the TableVectorizer performs properly when the
    type inferred during fit does not match the type of the
    data during transform.
    """
    table_vec = TableVectorizer(
        # only extract the total seconds
        datetime_transformer=DatetimeEncoder(resolution=None)
    )
    table_vec.fit(X_train)
    X_out = table_vec.transform(X_test)
    mask_nan = X_out == X_out
    assert_array_equal(X_out[mask_nan], expected_X_out[mask_nan])


def test_changing_types_int_float() -> None:
    """
    The TableVectorizer shouldn't cast floats to ints
    even if only ints were seen during fit
    """
    X_train = pd.DataFrame([[1], [2], [3]])
    X_test = pd.DataFrame([[1], [2], [3.3]])
    vectorizer = TableVectorizer().fit(X_train)
    X_trans = vectorizer.transform(X_test)
    expected_X_trans = np.array([[1.0], [2.0], [3.3]])
    assert_array_almost_equal(X_trans, expected_X_trans)


def test_column_by_column():
    """
    Test that the TableVectorizer gives the same result
    when applied column by column.
    """
    X = _get_clean_dataframe()
    vectorizer = TableVectorizer(
        high_cardinality_transformer=GapEncoder(n_components=2, random_state=0),
        cardinality_threshold=4,
    )
    vectorizer.set_output(transform="pandas")
    X_trans = vectorizer.fit_transform(X)
    feature_names = vectorizer.get_feature_names_out()
    for col in X.columns:
        X_trans_col = vectorizer.fit_transform(X[[col]])
        feature_names_filtered = [
            feat for feat in feature_names if feat.startswith(col)
        ]
        feature_names_col = vectorizer.get_feature_names_out()
        assert_array_equal(feature_names_col, feature_names_filtered)
        assert_array_equal(X_trans_col, X_trans[feature_names_col])


@skip_if_no_parallel
@pytest.mark.parametrize(
    "high_cardinality_transformer",
    # The GapEncoder and the MinHashEncoder should be parallelized on all columns.
    # The OneHotEncoder should not be parallelized.
    [
        GapEncoder(n_components=2, random_state=0),
        OneHotEncoder(),
        MinHashEncoder(n_components=2),
    ],
)
def test_parallelism(high_cardinality_transformer):
    X = _get_clean_dataframe()
    params = dict(
        high_cardinality_transformer=high_cardinality_transformer,
        cardinality_threshold=4,
    )
    vectorizer = TableVectorizer(**params)
    X_trans = vectorizer.fit_transform(X)

    with joblib.parallel_backend("loky"):
        for n_jobs in [None, 2, -1]:
            parallel_vectorizer = TableVectorizer(n_jobs=n_jobs, **params)
            X_trans_parallel = parallel_vectorizer.fit_transform(X)

            assert_array_equal(X_trans, X_trans_parallel)
            assert parallel_vectorizer.n_jobs == n_jobs

            # assert that all attributes are equal except for
            # the n_jobs attribute
            assert transformers_list_equal(
                parallel_vectorizer.transformers_,
                vectorizer.transformers_,
                ignore_params="n_jobs",
            )
            # assert that get_feature_names_out gives the same result
            assert_array_equal(
                vectorizer.get_feature_names_out(),
                parallel_vectorizer.get_feature_names_out(),
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
    assert table_vectorizer.named_transformers_["low_cardinality"].n_jobs is None

    table_vectorizer = TableVectorizer(
        numerical_transformer=DummyTransformerWithJobs(n_jobs=2),
        low_cardinality_transformer=DummyTransformerWithJobs(n_jobs=None),
        n_jobs=None,
    ).fit(X)
    assert table_vectorizer.named_transformers_["numeric"].n_jobs == 2
    assert table_vectorizer.named_transformers_["low_cardinality"].n_jobs is None

    # 2. Case where `TableVectorizer.n_jobs` is not `None` and we should propagate
    # when the underlying transformer `n_jobs` is not set explicitly.
    table_vectorizer = TableVectorizer(
        numerical_transformer=DummyTransformerWithJobs(n_jobs=None),
        low_cardinality_transformer=DummyTransformerWithJobs(n_jobs=None),
        n_jobs=2,
    ).fit(X)
    assert table_vectorizer.named_transformers_["numeric"].n_jobs == 2
    assert table_vectorizer.named_transformers_["low_cardinality"].n_jobs == 2

    # 3. Case where `TableVectorizer.n_jobs` is not `None` and we should not propagate
    # when the underlying transformer `n_jobs` is set explicitly.
    table_vectorizer = TableVectorizer(
        numerical_transformer=DummyTransformerWithJobs(n_jobs=4),
        low_cardinality_transformer=DummyTransformerWithJobs(n_jobs=None),
        n_jobs=2,
    ).fit(X)
    assert table_vectorizer.named_transformers_["numeric"].n_jobs == 4
    assert table_vectorizer.named_transformers_["low_cardinality"].n_jobs == 2


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


@pytest.mark.parametrize("invalid_transformer", [None, "drop"])
def test_wrong_transformer(invalid_transformer):
    X = _get_clean_dataframe()
    with pytest.raises(ValueError):
        TableVectorizer(high_cardinality_transformer=invalid_transformer).fit(X)


invalid_tuples = [
    (1, ValueError, r"(?=.*got scalar array)"),
    (np.array([1]), ValueError, r"(?=.*got 1D array)"),
    (pd.DataFrame([], columns=["a", "b"]), ValueError, r"(?=.*0 sample)"),
    (pd.DataFrame([], index=[0, 1]), ValueError, r"(?=.*0 feature)"),
    (pd.DataFrame([1], dtype="Sparse[int]"), TypeError, r"(?=.*sparse Pandas series)"),
    (csr_matrix([1]), TypeError, r"(?=.*A sparse matrix was passed)"),
]


@pytest.mark.parametrize("invalid_X, error, msg", invalid_tuples)
def test_invalid_X(invalid_X, error, msg):
    with pytest.raises(error, match=msg):
        TableVectorizer().fit(invalid_X)


def test_vectorize_datetime():
    X = pd.DataFrame({"A": [pd.Timestamp("2023-01-01", tz="UTC")]}).convert_dtypes()
    out = TableVectorizer().fit_transform(X)
    assert int(out[0, 0]) == 2023
