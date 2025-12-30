import re
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal, assert_raises
from pandas.testing import assert_frame_equal
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils._testing import skip_if_no_parallel
from sklearn.utils.fixes import parse_version

from skrub import _dataframe as sbd
from skrub._datetime_encoder import DatetimeEncoder
from skrub._gap_encoder import GapEncoder
from skrub._minhash_encoder import MinHashEncoder
from skrub._table_vectorizer import (
    Cleaner,
    TableVectorizer,
    _get_preprocessors,
)
from skrub._to_float import ToFloat
from skrub._to_str import ToStr
from skrub.conftest import _POLARS_INSTALLED

MSG_PANDAS_DEPRECATED_WARNING = "Skip deprecation warning"

PASSTHROUGH = FunctionTransformer(
    accept_sparse=True, check_inverse=False, feature_names_out="one-to-one"
)


def type_equality(expected_type, actual_type):
    """
    Checks that the expected type is equal to the actual type,
    assuming object and str types are equivalent
    (considered as categorical by the TableVectorizer).
    """
    if isinstance(expected_type, (object, str)) and isinstance(
        actual_type, (object, str)
    ):
        return True
    else:
        return expected_type == actual_type


def _get_clean_dataframe(df_module):
    """
    Creates a simple DataFrame with various types of data,
    and without missing values.
    """
    data1 = {
        "int": [15, 56, 63, 12, 44],
        "float": [5.2, 2.4, 6.2, 10.45, 9.0],
        "str1": ["public", "private", "private", "private", "public"],
        "str2": ["officer", "manager", "lawyer", "chef", "teacher"],
    }
    df1 = df_module.make_dataframe(data1)
    data2 = {
        "cat1": sbd.to_categorical(
            df_module.make_column("cat1", ["yes", "yes", "no", "yes", "no"])
        ),
        "cat2": sbd.to_categorical(
            df_module.make_column("cat2", ["20K+", "40K+", "60K+", "30K+", "50K+"])
        ),
    }
    df2 = df_module.make_dataframe(data2)
    return sbd.concat(df1, df2, axis=1)


def _get_dirty_dataframe(df_module, categorical_dtype="object"):
    data1 = {
        "int": [15, 56, None, 12, 44],
        "float": [5.2, 2.4, 6.2, 10.45, None],
    }
    df1 = df_module.make_dataframe(data1)

    # String and categorical values should be tested with both regular "string" and
    # "categorical" dtype. A separate dataframe is generated based on the case and
    # concatenated to the numeric features.
    if categorical_dtype == "category":
        data2 = {
            "str1": sbd.to_categorical(
                df_module.make_column(
                    "str1", ["public", None, "private", "private", "public"]
                )
            ),
            "str2": sbd.to_categorical(
                df_module.make_column(
                    "str2", ["officer", "manager", None, "chef", "teacher"]
                )
            ),
            "cat1": sbd.to_categorical(
                df_module.make_column("cat1", [None, "yes", "no", "yes", "no"])
            ),
            "cat2": sbd.to_categorical(
                df_module.make_column("cat2", ["20K+", "40K+", "60K+", "30K+", None])
            ),
        }
    else:
        data2 = {
            "str1": ["public", None, "private", "private", "public"],
            "str2": ["officer", "manager", None, "chef", "teacher"],
            "cat1": ["yes", "yes", "no", "yes", "no"],
            "cat2": ["20K+", "40K+", "60K+", "30K+", "50K+"],
        }
    df2 = df_module.make_dataframe(data2)
    return sbd.concat(df1, df2, axis=1)


def _get_mixed_types_dataframe(df_module):
    # TODO: This test should be modified so that it does not rely
    # on pd.NA
    data = {
        "int_str": ["1", "2", 3, "3", 5],
        "float_str": ["1.0", pd.NA, 3.0, "3.0", 5.0],
        "int_float": [1, 2, 3.0, 3, 5.0],
        "bool_str": ["True", False, True, "False", "True"],
    }
    return df_module.make_dataframe(data)


def _get_mixed_types_array():
    return np.array(
        [
            ["1", "2", 3, "3", 5],
            ["1.0", np.nan, 3.0, "3.0", 5.0],
            [1, 2, 3.0, 3, 5.0],
            ["True", False, True, "False", "True"],
        ]
    ).T


def _get_datetimes_dataframe(df_module):
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


# TODO: update this so it can generate polars dataframes
def _get_missing_values_dataframe(df_module, categorical_dtype="object"):
    """
    Creates a simple DataFrame with some columns that contain only missing values.
    We'll use different types of missing values (np.nan, pd.NA, None)
    to test how the vectorizer handles full null columns with mixed null values.
    """
    return df_module.make_dataframe(
        {
            "int": pd.Series([15, 56, pd.NA, 12, 44], dtype="Int64"),
            "all_null": pd.Series(
                [None, None, None, None, None], dtype=categorical_dtype
            ),
            "all_nan": pd.Series(
                [np.nan, np.nan, np.nan, np.nan, np.nan], dtype="Float64"
            ),
            "mixed_nulls": pd.Series(
                [np.nan, None, pd.NA, "NULL", "NA"], dtype=categorical_dtype
            ),
        }
    )


def test_get_preprocessors(df_module):
    X = _get_clean_dataframe(df_module)
    steps = _get_preprocessors(
        cols=X.columns,
        drop_null_fraction=1.0,
        drop_if_constant=True,
        drop_if_unique=False,
        n_jobs=1,
        add_tofloat32=True,
    )
    assert any(isinstance(step.transformer, ToFloat) for step in steps[1:])

    steps = _get_preprocessors(
        cols=X.columns,
        drop_null_fraction=1.0,
        drop_if_constant=True,
        drop_if_unique=False,
        n_jobs=1,
        add_tofloat32=False,
    )
    assert not any(isinstance(step.transformer, ToFloat) for step in steps[1:])


def test_fit_default_transform(df_module):
    X = _get_clean_dataframe(df_module)
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


def passthrough_vectorizer():
    return TableVectorizer(
        high_cardinality="passthrough",
        low_cardinality="passthrough",
        numeric="passthrough",
        datetime="passthrough",
    )


@pytest.mark.parametrize(
    "data_getter, expected_types",
    [
        (
            _get_datetimes_dataframe,
            {
                "pd_datetime": "datetime",
                "np_datetime": "datetime",
                "dmy-": "datetime",
                "ymd/": "datetime",
                "ymd/_hms:": "datetime",
            },
        ),
        (
            _get_clean_dataframe,
            {
                "int": "float32",
                "float": "float32",
                "str1": "string",
                "str2": "string",
                "cat1": "category",
                "cat2": "category",
            },
        ),
        (
            lambda df_module: _get_dirty_dataframe(df_module, "category"),
            {
                "int": "float32",
                "float": "float32",
                "str1": "category",
                "str2": "category",
                "cat1": "category",
                "cat2": "category",
            },
        ),
    ],
)
def test_auto_cast(data_getter, expected_types, df_module):
    """
    Tests that the TableVectorizer automatic type detection works as expected.
    """
    X = data_getter(df_module)
    vectorizer = passthrough_vectorizer()
    X_trans = vectorizer.fit_transform(X)
    for col in X_trans.columns:
        if expected_types[col] == "datetime":
            assert sbd.is_any_date(X_trans[col])
        if expected_types[col] == "category":
            assert sbd.is_categorical(X_trans[col])
        if expected_types[col] == "float32":
            assert sbd.is_float(X_trans[col])
        if expected_types[col] == "string":
            assert sbd.is_string(X_trans[col])


@pytest.mark.parametrize(
    "data_getter, expected_types",
    [
        (
            _get_datetimes_dataframe,
            {
                "pd_datetime": "datetime",
                "np_datetime": "datetime",
                "dmy-": "datetime",
                "ymd/": "datetime",
                "ymd/_hms:": "datetime",
            },
        ),
        (
            _get_clean_dataframe,
            {
                "int": "int",
                "float": "float",
                "str1": "string",
                "str2": "string",
                "cat1": "category",
                "cat2": "category",
            },
        ),
        (
            lambda df_module: _get_dirty_dataframe(df_module, "category"),
            {
                "int": "int",
                "float": "float",
                "str1": "category",
                "str2": "category",
                "cat1": "category",
                "cat2": "category",
            },
        ),
    ],
)
def test_cleaner_dtypes(data_getter, expected_types, df_module):
    X = data_getter(df_module)
    # datetimes dataframe does not contain int cols
    if "_get_datetimes_dataframe" not in str(data_getter):
        if df_module.description == "pandas-numpy-dtypes" and sbd.has_nulls(X["int"]):
            # Numpy dtypes fail when an integer column contains a null value, so we
            # skip this test
            pytest.xfail(
                reason=(
                    "Test is expected to fail for dirty dataframe with"
                    " pandas-numpy-dtypes configuration"
                ),
            )
    vectorizer = Cleaner()
    X_trans = vectorizer.fit_transform(X)
    for col in X_trans.columns:
        if expected_types[col] == "datetime":
            assert sbd.is_any_date(X_trans[col])
        else:
            for col, dtype in expected_types.items():
                if dtype == "int":
                    assert sbd.is_integer(X_trans[col])
                elif dtype == "float":
                    assert sbd.is_float(X_trans[col])
                elif dtype == "category":
                    assert sbd.is_categorical(X_trans[col])
                else:  # string
                    assert sbd.is_string(X_trans[col])

    X_trans = vectorizer.fit(X).transform(X)
    for col in X_trans.columns:
        if expected_types[col] == "datetime":
            assert sbd.is_any_date(X_trans[col])
        else:
            for col, dtype in expected_types.items():
                if dtype == "int":
                    assert sbd.is_integer(X_trans[col])
                elif dtype == "float":
                    assert sbd.is_float(X_trans[col])
                elif dtype == "category":
                    assert sbd.is_categorical(X_trans[col])
                else:  # string
                    assert sbd.is_string(X_trans[col])


def test_convert_float32(df_module):
    """
    Test that the TableVectorizer converts float64 to float32
    when using the default parameters.
    """
    X = _get_clean_dataframe(df_module)
    vectorizer = TableVectorizer()
    out = vectorizer.fit_transform(X)
    # this syntax is needed because polars dtypes aren't represented as strings
    assert sbd.dtype(out["float"]) == sbd.dtype(sbd.to_float32(X["float"]))
    assert sbd.dtype(out["int"]) == sbd.dtype(sbd.to_float32(X["int"]))

    # default behavior: keep numeric type
    vectorizer = Cleaner()
    out = vectorizer.fit_transform(X)
    # here we don't need to convert because we're just checking that the output
    # dtypes are the same as the input dtypes
    assert sbd.dtype(out["float"]) == sbd.dtype(X["float"])
    assert sbd.dtype(out["int"]) == sbd.dtype(X["int"])

    vectorizer = Cleaner(numeric_dtype="float32")
    out = vectorizer.fit_transform(X)
    # here it's the same as the case with the TableVectorizer above
    assert sbd.dtype(out["float"]) == sbd.dtype(sbd.to_float32(X["float"]))
    assert sbd.dtype(out["int"]) == sbd.dtype(sbd.to_float32(X["int"]))


def test_cast_to_str(df_module):
    """
    Test that the Cleaner conditionally applies the ToStr transformer
    depending on the cast_to_str parameter.
    """
    df = df_module.DataFrame({"a": [[1, 2], [3]]})

    # -----------------------
    # Case 0: default (False)
    # -----------------------
    cleaner = Cleaner()
    out = cleaner.fit_transform(df)
    # Default should preserve the dtype
    assert sbd.dtype(out["a"]) == sbd.dtype(df["a"])

    # -----------------------
    # Case 1: cast_to_str=False
    # -----------------------
    cleaner = Cleaner(cast_to_str=False)
    out = cleaner.fit_transform(df)

    # Should preserve dtype
    assert sbd.dtype(out["a"]) == sbd.dtype(df["a"])

    # ----------------------
    # Case 2: cast_to_str=True
    # ----------------------
    cleaner = Cleaner(cast_to_str=True)
    out = cleaner.fit_transform(df)

    expected_col = ToStr().fit_transform(df["a"])
    assert sbd.dtype(out["a"]) == sbd.dtype(expected_col)


def test_cleaner_invalid_numeric_dtype(df_module):
    X = _get_clean_dataframe(df_module)
    with pytest.raises(ValueError, match="numeric_dtype.*must be one of"):
        Cleaner(numeric_dtype="wrong").fit_transform(X)


def test_cleaner_get_feature_names_out(df_module):
    """Test that Cleaner.get_feature_names_out returns the correct column names."""
    X = _get_clean_dataframe(df_module)
    cleaner = Cleaner().fit(X)

    # Get feature names from the cleaner
    feature_names = cleaner.get_feature_names_out()

    # Get the expected column names from the transformed dataframe
    X_transformed = cleaner.transform(X)
    expected_names = sbd.column_names(X_transformed)

    # Check that they match
    assert_array_equal(feature_names, expected_names)

    # Test that it works with drop_null_fraction parameter
    X_with_nulls = df_module.make_dataframe(
        {
            "col1": [1, 2, 3, 4],
            "col2": ["a", "b", "c", "d"],
            "all_nulls": [None, None, None, None],
        }
    )

    cleaner_drop = Cleaner(drop_null_fraction=1.0).fit(X_with_nulls)
    feature_names_drop = cleaner_drop.get_feature_names_out()

    # The all_nulls column should not be in the output
    assert "all_nulls" not in feature_names_drop

    # Test that input_features parameter is ignored (like in TableVectorizer)
    feature_names_with_input = cleaner.get_feature_names_out(
        input_features=["dummy1", "dummy2"]
    )
    assert_array_equal(feature_names_with_input, expected_names)


def test_auto_cast_missing_categories(df_module):
    # TODO implement test for polars
    if df_module.description == "polars":
        pytest.skip("Skipping test for polars")

    X = _get_dirty_dataframe(df_module, "category")
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

    X = _get_dirty_dataframe(df_module, "category")
    X_train = X.head(3).reset_index(drop=True)
    X_test = X.tail(2).reset_index(drop=True)
    _ = vectorizer.fit_transform(X_train)
    out = vectorizer.transform(X_test)

    assert dict(out.dtypes) == expected_type_per_column


def test_get_feature_names_out(df_module):
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
    X = _get_clean_dataframe(df_module)
    rng = np.random.default_rng(42)
    y = rng.integers(0, 2, size=X.shape[0])
    vectorizer = TableVectorizer().fit(X)
    assert_array_equal(
        vectorizer.get_feature_names_out(),
        expected_features,
    )

    # make sure that `get_feature_names` works when `TableVectorizer` is used in a
    # scikit-learn pipeline
    # non-regression test for https://github.com/skrub-data/skrub/issues/1256
    pipeline = make_pipeline(TableVectorizer(), StandardScaler(), Ridge()).fit(X, y)
    assert_array_equal(
        pipeline[:-1].get_feature_names_out(),
        expected_features,
    )
    # Input features are ignored when `TableVectorizer` is used in a pipeline
    assert_array_equal(
        pipeline[:-1].get_feature_names_out(
            input_features=[f"col_{i}" for i in range(X.shape[1])]
        ),
        expected_features,
    )


def test_transform(df_module):
    X = _get_clean_dataframe(df_module)
    table_vec = TableVectorizer().fit(X)

    x = {
        "int": [34],
        "float": [5.5],
        "str1": ["private"],
        "str2": ["manager"],
        "cat1": ["yes"],
        "cat2": ["60K+"],
    }
    x = df_module.make_dataframe(x)

    x_trans = table_vec.transform(x)
    expected_x_trans = [
        [34.0, 5.5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    ]
    assert_array_equal(x_trans, expected_x_trans)


@pytest.mark.parametrize(
    "data_getter",
    [
        _get_clean_dataframe,
        # lambda functions are needed to access df_module
        lambda df_module: _get_dirty_dataframe(df_module, categorical_dtype="object"),
        lambda df_module: _get_dirty_dataframe(df_module, categorical_dtype="category"),
        _get_mixed_types_dataframe,
        lambda df_module: _get_mixed_types_array(),
    ],
)
def test_fit_transform_equiv(data_getter, df_module):
    # TODO: this check is probably already performed in test_sklearn
    """
    We will test the equivalence between using `.fit_transform(X)`
    and `.fit(X).transform(X).`
    """
    # data_getter may be an array, otherwise call data_getter and get the result
    X = data_getter(df_module) if callable(data_getter) else data_getter
    X_trans_1 = TableVectorizer().fit_transform(X)
    X_trans_2 = TableVectorizer().fit(X).transform(X)
    assert_array_equal(X_trans_1, X_trans_2)


def test_handle_unknown_category(df_module):
    X = _get_clean_dataframe(df_module)
    # Treat all columns as having few unique values
    table_vec = TableVectorizer(cardinality_threshold=7).fit(X)
    data_unknown = {
        "int": [3, 1],
        "float": [2.1, 4.3],
        "str1": ["semi-private", "public"],
        "str2": ["researcher", "chef"],
        "cat1": ["maybe", "yes"],
        "cat2": ["70K+", "20K+"],
    }
    X_unknown = df_module.make_dataframe(data_unknown)

    data_known = {
        "int": [1, 4],
        "float": [4.3, 3.3],
        "str1": ["public", "private"],
        "str2": ["chef", "chef"],
        "cat1": ["yes", "no"],
        "cat2": ["30K+", "20K+"],
    }
    X_known = df_module.make_dataframe(data_known)

    # Default behavior is "handle_unknown='ignore'",
    # so unknown categories are encoded as all zeros
    X_trans_unknown = table_vec.transform(X_unknown)
    X_trans_known = table_vec.transform(X_known)

    assert X_trans_unknown.shape == X_trans_known.shape

    # +2 for binary columns which get one category dropped
    n_zeroes = sbd.n_unique(X["str2"]) + sbd.n_unique(X["cat2"]) + 2

    # This is a convoluted syntax for checking that all the columns from the
    # second to the last are empty (because they're unknown categories)
    colnames = sbd.column_names(X_trans_unknown)[2:n_zeroes]
    assert_array_equal(
        sbd.slice(X_trans_unknown[colnames], 0, 1).to_numpy().squeeze(),
        np.zeros_like(X_trans_unknown)[0, 2:n_zeroes],
    )
    assert_raises(
        AssertionError,
        assert_array_equal,
        sbd.slice(X_trans_known, 0, 1).to_numpy().squeeze(),
        np.zeros_like(X_trans_unknown)[0, :n_zeroes],
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
def test_deterministic(pipeline, df_module):
    """
    Tests that running the same TableVectorizer multiple times with the same
    (deterministic) components results in the same output.
    """
    X = _get_dirty_dataframe(df_module)
    X_trans_1 = pipeline.fit_transform(X)
    X_trans_2 = pipeline.fit_transform(X)
    assert_array_equal(X_trans_1, X_trans_2)


def test_mixed_types(df_module):
    """
    Check that the types are correctly inferred.
    """
    if parse_version(pd.__version__) < parse_version("2.0.0"):
        pytest.xfail("pandas is_string_dtype incorrect in old pandas")
    X = _get_mixed_types_dataframe(df_module)
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
        datetime=DatetimeEncoder(resolution=None),
        drop_null_fraction=None,
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


def test_column_by_column(df_module):
    """
    Test that the TableVectorizer gives the same result
    when applied column by column.
    """
    if parse_version(pd.__version__) < parse_version("2.0.0"):
        pytest.xfail("pandas is_string_dtype incorrect in old pandas")
    X = _get_clean_dataframe(df_module)
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
def test_parallelism(high_cardinality, df_module):
    X = _get_clean_dataframe(df_module)
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


def test_wrong_transformer(df_module):
    X = _get_clean_dataframe(df_module)
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


def test_sk_visual_block(df_module):
    X = _get_clean_dataframe(df_module)
    vectorizer = TableVectorizer()
    unfitted_repr = vectorizer._repr_html_()
    assert "TableVectorizer" in unfitted_repr
    vectorizer.fit(X)
    assert (
        "[&#x27;str1&#x27;, &#x27;str2&#x27;, &#x27;cat1&#x27;, &#x27;cat2&#x27;]"
        in vectorizer._repr_html_()
    )


def test_supervised_encoder(df_module):
    TargetEncoder = pytest.importorskip("sklearn.preprocessing.TargetEncoder")
    # test that the vectorizer works correctly with encoders that need y (none
    # of the defaults encoders do)
    X = df_module.make_dataframe({"a": [f"c_{i}" for _ in range(5) for i in range(4)]})
    y = np.random.default_rng(0).normal(size=sbd.shape(X)[0])
    tv = TableVectorizer(low_cardinality=TargetEncoder())
    tv.fit_transform(X, y)


def test_drop_null_column(df_module):
    """Check that all null columns are dropped, and no more."""
    # TODO: avoid skipping by adding proper polars support to
    # _get_missing_values_dataframe
    pytest.importorskip("pyarrow")
    X = _get_missing_values_dataframe(df_module)
    # Don't drop null columns
    tv = TableVectorizer(drop_null_fraction=None)
    transformed = tv.fit_transform(X)

    assert sbd.shape(transformed) == sbd.shape(X)

    # Drop null columns
    tv = TableVectorizer(drop_null_fraction=1.0)
    transformed = tv.fit_transform(X)
    assert sbd.shape(transformed) == (sbd.shape(X)[0], 1)


def test_date_format(df_module):
    # Test that the date format is correctly inferred

    X = df_module.make_dataframe(
        {
            "date": [
                "22 April 2025",
                "23 April 2025",
                "24 April 2025",
                "25 April 2025",
                "26 April 2025",
            ]
        }
    )

    expected = df_module.make_dataframe(
        {
            "date_year": [2025.0, 2025.0, 2025.0, 2025.0, 2025.0],
            "date_month": [4.0, 4.0, 4.0, 4.0, 4.0],
            "date_day": [22.0, 23.0, 24.0, 25.0, 26.0],
        }
    )
    datetime_encoder = DatetimeEncoder(add_total_seconds=False)
    vectorizer = TableVectorizer(datetime_format="%d %B %Y", datetime=datetime_encoder)
    transformed = vectorizer.fit_transform(X)
    for col in transformed.columns:
        df_module.assert_column_equal(transformed[col], sbd.to_float32(expected[col]))

    expected = df_module.make_dataframe(
        {
            "date": [
                datetime.fromisoformat("2025-04-22"),
                datetime.fromisoformat("2025-04-23"),
                datetime.fromisoformat("2025-04-24"),
                datetime.fromisoformat("2025-04-25"),
                datetime.fromisoformat("2025-04-26"),
            ],
        }
    )
    cleaner = Cleaner(datetime_format="%d %B %Y")
    transformed = cleaner.fit_transform(X)

    # This is needed because the transformed version of the date column is formatted
    # by pandas using the format, and has a different resolution than the expected one.
    transformed_to_list = sbd.to_list(transformed["date"])
    expected_to_list = sbd.to_list(expected["date"])
    assert transformed_to_list == expected_to_list


@pytest.mark.skipif(
    not _POLARS_INSTALLED,
    reason="This test requires polars to be installed",
)
def test_cleaner_empty_column_name():
    import polars as pl

    # non-regression test for issue https://github.com/skrub-data/skrub/issues/1490
    df = pl.DataFrame({"": [1], "b": [2], "c": [""]})
    cleaner = Cleaner()
    cleaner.fit_transform(df)
    assert list(cleaner.all_processing_steps_.keys()) == df.columns
    assert all(len(step) > 0 for step in cleaner.all_processing_steps_.values())
