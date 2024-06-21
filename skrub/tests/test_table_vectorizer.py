import re
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_raises
from pandas.testing import assert_frame_equal
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.utils._testing import skip_if_no_parallel
from sklearn.utils.fixes import parse_version

from skrub import _dataframe as sbd
from skrub._datetime_encoder import DatetimeEncoder
from skrub._gap_encoder import GapEncoder
from skrub._minhash_encoder import MinHashEncoder
from skrub._table_vectorizer import TableVectorizer

MSG_PANDAS_DEPRECATED_WARNING = "Skip deprecation warning"

if parse_version(sklearn.__version__) < parse_version("1.4"):
    PASSTHROUGH = "passthrough"
else:
    PASSTHROUGH = FunctionTransformer(
        accept_sparse=True, check_inverse=False, feature_names_out="one-to-one"
    )


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
        }
    )


def test_fit_default_transform():
    X = _get_clean_dataframe()
    vectorizer = TableVectorizer()
    vectorizer.fit(X)

    low_cardinality_cols = ["str1", "str2", "cat1", "cat2"]
    expected_transformers_types = {}
    for c in X.columns:
        if c in ["int", "float"]:
            expected_transformers_types[c] = "PassThrough"
        elif c in low_cardinality_cols:
            expected_transformers_types[c] = "OneHotEncoder"
        else:
            expected_transformers_types[c] = "GapEncoder"

    transformer_types = {
        k: v.__class__.__name__ for (k, v) in vectorizer.transformers_.items()
    }
    assert transformer_types == expected_transformers_types
    categories = vectorizer.transformers_["cat1"].categories_[0]
    expected_categories = ["no", "yes"]
    assert list(categories) == list(expected_categories)


def test_duplicate_column_names():
    """
    Test to check if the tablevectorizer raises an error with
    duplicate column names
    """
    tablevectorizer = TableVectorizer()
    # Creates a simple dataframe with duplicate column names
    X_dup_col_names = pd.DataFrame(np.ones((3, 2)), columns=["col_1", "col_1"])

    with pytest.warns(UserWarning, match=".*duplicated column names.*"):
        transformed = tablevectorizer.fit_transform(X_dup_col_names)
    cols = list(transformed.columns)
    assert len(cols) == 2
    assert cols[0] == "col_1"
    assert re.match(r"^col_1__skrub_[0-9a-f]+__$", cols[1])


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
        },
    ),
    # Test other types detection
    (
        _get_clean_dataframe(),
        {
            "int": "float32",
            "float": "float32",
            "str1": "O",
            "str2": "O",
            "cat1": "category",
            "cat2": "category",
        },
    ),
    (
        _get_dirty_dataframe("category"),
        {
            "int": "float32",
            "float": "float32",
            "str1": "category",
            "str2": "category",
            "cat1": "category",
            "cat2": "category",
        },
    ),
]


def passthrough_vectorizer():
    return TableVectorizer(
        high_cardinality="passthrough",
        low_cardinality="passthrough",
        numeric="passthrough",
        datetime="passthrough",
    )


@pytest.mark.parametrize("X, dict_expected_types", X_tuples)
def test_auto_cast(X, dict_expected_types):
    """
    Tests that the TableVectorizer automatic type detection works as expected.
    """
    vectorizer = passthrough_vectorizer()
    X_trans = vectorizer.fit_transform(X)
    for col in X_trans.columns:
        assert dict_expected_types[col] == X_trans[col].dtype


def test_auto_cast_missing_categories():
    X = _get_dirty_dataframe("category")
    vectorizer = passthrough_vectorizer()
    out = vectorizer.fit_transform(X)

    expected_type_per_column = {
        "int": "float32",
        "float": "float32",
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
    assert dict(out.dtypes) == expected_type_per_column

    X = _get_dirty_dataframe("category")
    X_train = X.head(3).reset_index(drop=True)
    X_test = X.tail(2).reset_index(drop=True)
    _ = vectorizer.fit_transform(X_train)
    out = vectorizer.transform(X_test)

    assert dict(out.dtypes) == expected_type_per_column


def test_get_feature_names_out():
    expected_features = [
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
    X = _get_clean_dataframe()
    vectorizer = TableVectorizer().fit(X)
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


def test_handle_unknown_category():
    X = _get_clean_dataframe()
    # Treat all columns as having few unique values
    table_vec = TableVectorizer(cardinality_threshold=7).fit(X)
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
        X_trans_unknown.iloc[0, 2:n_zeroes],
        np.zeros_like(X_trans_unknown.iloc[0, 2:n_zeroes]),
    )
    assert_raises(
        AssertionError,
        assert_array_equal,
        X_trans_known.iloc[0, :n_zeroes],
        np.zeros_like(X_trans_known.iloc[0, :n_zeroes]),
    )


@pytest.mark.parametrize(
    "pipeline",
    [
        TableVectorizer(),
        TableVectorizer(
            low_cardinality=MinHashEncoder(),
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
    if parse_version(pd.__version__) < parse_version("2.0.0"):
        pytest.xfail("pandas is_string_dtype incorrect in old pandas")
    X = _get_mixed_types_dataframe()
    vectorizer = TableVectorizer()
    vectorizer.fit(X)
    expected_transformer_types = {
        "int_str": "PassThrough",
        "float_str": "PassThrough",
        "int_float": "PassThrough",
        "bool_str": "OneHotEncoder",
    }
    transformer_types = {
        k: v.__class__.__name__ for k, v in vectorizer.transformers_.items()
    }
    assert expected_transformer_types == transformer_types


@pytest.mark.parametrize(
    "X_train, X_test, expected_X_out",
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
            pd.DataFrame({"col1_total_seconds": [1.5463008e09, 1.5463872e09, np.nan]}),
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
        datetime=DatetimeEncoder(resolution=None)
    )
    table_vec.fit(X_train)
    X_out = table_vec.transform(X_test)
    assert (X_out.isna() == expected_X_out.isna()).all().all()
    assert (X_out.dropna() == expected_X_out.dropna()).all().all()


def test_changing_types_int_float():
    """
    The TableVectorizer shouldn't cast floats to ints
    even if only ints were seen during fit.
    """
    X_train = pd.DataFrame([[1.1], [2], [3]], columns=["a"])
    X_test = pd.DataFrame([[1], [2], [3.3]], columns=["a"])
    vectorizer = TableVectorizer().fit(X_train)
    X_trans = vectorizer.transform(X_test)
    expected_X_trans = np.array([[1.0], [2.0], [3.3]])
    assert_array_almost_equal(X_trans, expected_X_trans)


def test_column_by_column():
    """
    Test that the TableVectorizer gives the same result
    when applied column by column.
    """
    if parse_version(pd.__version__) < parse_version("2.0.0"):
        pytest.xfail("pandas is_string_dtype incorrect in old pandas")
    X = _get_clean_dataframe()
    vectorizer = TableVectorizer(
        high_cardinality=GapEncoder(n_components=2, random_state=0),
        cardinality_threshold=4,
    )
    X_trans = vectorizer.fit_transform(X)
    for col in X.columns:
        col_vect = clone(vectorizer)
        X_trans_col = col_vect.fit_transform(X[[col]])
        feature_names_filtered = vectorizer.input_to_outputs_[col]
        feature_names_col = col_vect.get_feature_names_out()
        assert_array_equal(feature_names_col, feature_names_filtered)
        assert_array_equal(X_trans_col, X_trans[feature_names_col])


@skip_if_no_parallel
@pytest.mark.parametrize(
    "high_cardinality",
    # The GapEncoder and the MinHashEncoder should be parallelized on all columns.
    # The OneHotEncoder should not be parallelized.
    [
        GapEncoder(n_components=2, random_state=0),
        OneHotEncoder(sparse_output=False),
        MinHashEncoder(n_components=2),
    ],
)
def test_parallelism(high_cardinality):
    X = _get_clean_dataframe()
    params = dict(
        high_cardinality=high_cardinality,
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

            # assert that get_feature_names_out gives the same result
            assert_array_equal(
                vectorizer.get_feature_names_out(),
                parallel_vectorizer.get_feature_names_out(),
            )


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


def test_wrong_transformer():
    X = _get_clean_dataframe()
    with pytest.raises(ValueError):
        TableVectorizer(high_cardinality="passthroughtypo").fit(X)
    with pytest.raises(TypeError):
        TableVectorizer(high_cardinality=None).fit(X)


invalid_tuples = [
    (1, TypeError, r".*Only pandas and polars DataFrames.*"),
    (np.array([1]), ValueError, r".*incompatible shape.*"),
    (pd.DataFrame([1], dtype="Sparse[int]"), TypeError, r".*sparse Pandas series.*"),
    (pd.Series([1, 2], name="S"), TypeError, r".*Only pandas and polars DataFrames.*"),
    (csr_matrix([1]), TypeError, r".*Only pandas and polars DataFrames.*"),
]


@pytest.mark.parametrize("invalid_X, error, msg", invalid_tuples)
def test_invalid_X(invalid_X, error, msg):
    try:
        # avoid warning
        invalid_X.columns = list(map(str, invalid_X.columns))
    except AttributeError:
        pass
    with pytest.raises(error, match=msg):
        TableVectorizer().fit(invalid_X)


def test_vectorize_datetime():
    X = pd.DataFrame({"A": [pd.Timestamp("2023-01-01", tz="UTC")]}).convert_dtypes()
    out = TableVectorizer().fit_transform(X)
    assert int(out.iloc[0, 0]) == 2023


def test_specific_transformers():
    df = pd.DataFrame(dict(a1=[1, 2, 3], a2=[1, 2, 3], b1=["a", "b", "c"]))
    tv = TableVectorizer(
        specific_transformers=[
            (
                FunctionTransformer(lambda df: df.rename(columns=lambda c: f"{c}_new")),
                ["b1"],
            ),
            (FunctionTransformer(lambda df: df * 2), ["a2"]),
        ]
    )
    out = tv.fit_transform(df)
    expected = pd.DataFrame(
        dict(
            a1=pd.Series([1.0, 2.0, 3.0], dtype="float32"),
            a2=[2, 4, 6],
            b1_new=["a", "b", "c"],
        )
    )
    assert_frame_equal(out, expected)
    with pytest.raises(ValueError, match=".*twice"):
        TableVectorizer(
            specific_transformers=[
                ("passthrough", ["a1", "b1"]),
                ("passthrough", ["a1"]),
            ]
        ).fit_transform(df)

    with pytest.raises(
        ValueError, match="'specific_transformers' must be a list .* pairs"
    ):
        TableVectorizer(
            specific_transformers=[("name", "passthrough", ["a1", "b1"])]
        ).fit_transform(df)


def test_accept_pipeline():
    # non-regression test for https://github.com/skrub-data/skrub/issues/886
    # TableVectorizer used to force transformers to inherit from TransformerMixin
    df = pd.DataFrame(dict(a=[1.1, 2.2]))
    tv = TableVectorizer(numeric=make_pipeline("passthrough"))
    tv.fit(df)


@pytest.mark.skipif(
    parse_version(sklearn.__version__) < parse_version("1.4"),
    reason="set_output('polars') was added in scikit-learn 1.4",
)
def test_clean_null_downcast_warning():
    # non-regression test for https://github.com/skrub-data/skrub/issues/894
    pl = pytest.importorskip("polars")
    df = pl.DataFrame(dict(a=[0, 1], b=["a", "b"]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TableVectorizer().fit_transform(df)


def test_numberlike_categories():
    # non-regression test for https://github.com/skrub-data/skrub/issues/874
    # TableVectorizer would not apply the same transformations in fit and
    # transform and end up treating numbers as categories
    df = pd.DataFrame(dict(a=pd.Series(["0", "1"], dtype="category")))
    TableVectorizer().fit(df).transform(df)


def test_bad_specific_cols():
    with pytest.raises(
        ValueError, match=".* must be a list of .transformer, list of columns."
    ):
        TableVectorizer(specific_transformers=[(None, "a")]).fit(None)
    with pytest.raises(
        ValueError, match="Column names in 'specific_transformers' must be strings"
    ):
        TableVectorizer(specific_transformers=[(None, [0])]).fit(None)


def test_supervised_encoder(df_module):
    TargetEncoder = pytest.importorskip("sklearn.preprocessing.TargetEncoder")
    # test that the vectorizer works correctly with encoders that need y (none
    # of the defaults encoders do)
    X = df_module.make_dataframe({"a": [f"c_{i}" for _ in range(5) for i in range(4)]})
    y = np.random.default_rng(0).normal(size=sbd.shape(X)[0])
    tv = TableVectorizer(low_cardinality=TargetEncoder())
    tv.fit_transform(X, y)
