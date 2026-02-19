import datetime

import numpy as np
import pytest
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from skrub import ApplyToCols, ApplyToEachCol
from skrub import _dataframe as sbd
from skrub import selectors as s
from skrub._apply_sub_frame import ApplyToSubFrame
from skrub._to_datetime import ToDatetime


def test_single_column_transformer_becomes_apply_to_each_col(df_module):
    """SingleColumnTransformer should be wrapped as ApplyToEachCol."""
    at = ApplyToCols(ToDatetime(), cols=s.all())
    X = df_module.make_dataframe({"date_col": ["2020-01-01", "2020-01-02"]})
    at.fit(X)
    assert isinstance(at._wrapped_transformer, ApplyToEachCol)


def test_non_single_column_transformer_becomes_apply_to_subframe(df_module):
    """Non-SingleColumnTransformer should be wrapped as ApplyToSubFrame."""
    at = ApplyToCols(OrdinalEncoder(), cols=s.all())
    X = df_module.make_dataframe({"col1": ["a", "b"], "col2": ["x", "y"]})
    at.fit(X)
    assert isinstance(at._wrapped_transformer, ApplyToSubFrame)


def test_columnwise_override_forces_apply_to_each_col(df_module):
    """
    columnwise=True should force ApplyToEachCol even
    for non-SingleColumnTransformer.
    """
    at = ApplyToCols(OrdinalEncoder(), cols=s.all(), columnwise=True)
    X = df_module.make_dataframe({"col1": ["a", "b"], "col2": ["x", "y"]})
    at.fit(X)
    assert isinstance(at._wrapped_transformer, ApplyToEachCol)


def test_invalid_parameters():
    """all these parameters should be boolean."""

    X = None  # Placeholder for the dataframe, not used in this test

    with pytest.raises((TypeError, ValueError)):
        at = ApplyToCols(ToDatetime(), allow_reject="yes")
        at.fit_transform(X)
    with pytest.raises((TypeError, ValueError)):
        at = ApplyToCols(ToDatetime(), keep_original="no")
        at.fit_transform(X)
    with pytest.raises((TypeError, ValueError)):
        at = ApplyToCols(ToDatetime(), columnwise="maybe")
        at.fit_transform(X)


def test_to_datetime_transformation(df_module):
    """Test transformation with ToDatetime (SingleColumnTransformer)."""
    at = ApplyToCols(ToDatetime(), cols=s.all(), allow_reject=True)
    X = df_module.make_dataframe(
        {"date": ["2020-01-01", "2020-01-02"], "value": [1, 2]}
    )
    X_transformed = at.fit_transform(X)

    # Check that date column was transformed to datetime
    assert sbd.is_any_date(sbd.col(X_transformed, "date"))
    # Check that value column was unchanged
    assert sbd.to_list(sbd.col(X_transformed, "value")) == [1, 2]


def test_ordinal_encoder_transformation(df_module):
    """Test transformation with OrdinalEncoder (non-SingleColumnTransformer)."""
    at = ApplyToCols(OrdinalEncoder(), cols=s.all())
    X = df_module.make_dataframe({"col1": ["a", "b", "c"], "col2": ["x", "y", "z"]})
    X_transformed = at.fit_transform(X)

    # Check that data was encoded properly
    assert np.array_equal(sbd.col(X_transformed, "col1"), [0, 1, 2])
    assert np.array_equal(sbd.col(X_transformed, "col2"), [0, 1, 2])


def test_column_selection_with_selector(df_module):
    """Test that only selected columns are transformed."""
    at = ApplyToCols(OrdinalEncoder(), cols=s.string())
    X = df_module.make_dataframe(
        {
            "numeric1": [1.0, 2.0, 3.0],
            "numeric2": [10.0, 20.0, 30.0],
            "string_col": ["a", "b", "c"],
        }
    )
    X_transformed = at.fit_transform(X)

    # Check that only string_col was transformed
    assert np.array_equal(sbd.col(X_transformed, "string_col"), [0, 1, 2])
    # Check that numeric columns were unchanged
    assert np.array_equal(sbd.col(X_transformed, "numeric1"), [1.0, 2.0, 3.0])
    assert np.array_equal(sbd.col(X_transformed, "numeric2"), [10.0, 20.0, 30.0])


def test_fit_and_transform_separate(df_module):
    """Test that fit and transform can be called separately."""
    at = ApplyToCols(OrdinalEncoder(), cols=s.string())
    X_train = df_module.make_dataframe(
        {"col1": ["a", "b", "c"], "col2": ["x", "y", "z"]}
    )
    X_test = df_module.make_dataframe({"col1": ["a", "b"], "col2": ["x", "y"]})

    at.fit(X_train)
    X_train_transformed = at.transform(X_train)
    X_test_transformed = at.transform(X_test)

    assert sbd.shape(X_train_transformed) == (3, 2)
    assert sbd.shape(X_test_transformed) == (2, 2)


def test_correct_transformer_attributes(df_module):
    """Test that transformer attributes are correctly set after fit."""
    at = ApplyToCols(StandardScaler(), cols=s.numeric())
    X = df_module.make_dataframe(
        {
            "col1": [1.0, 2.0, 3.0],
            "col2": [10.0, 20.0, 30.0],
            "date_col": ["2020-01-01", "2020-01-02", "2020-01-03"],
        }
    )
    at.fit(X)

    # For non-SingleColumnTransformer, transformer_ should be set
    assert hasattr(at, "transformer_")
    assert isinstance(at.transformer_, StandardScaler)

    # For SingleColumnTransformer, transformers_ should be set
    at_single = ApplyToCols(ToDatetime(), cols="date_col")
    at_single.fit(X)
    assert hasattr(at_single, "transformers_")
    assert isinstance(at_single.transformers_, dict)


def test_reject_column(df_module):
    at = ApplyToCols(ToDatetime(), cols=s.all(), allow_reject=True)
    X = df_module.make_dataframe(
        {"date": ["2020-01-01", "2020-01-02"], "value": [1, 2]}
    )
    X_transformed = at.fit_transform(X)

    X_expected = df_module.make_dataframe(
        {
            "date": [datetime.datetime(2020, 1, 1), datetime.datetime(2020, 1, 2)],
            "value": [1, 2],
        }
    )

    df_module.assert_frame_equal(X_transformed, X_expected)

    # A RejectColumn exception should be raised
    with pytest.raises(ValueError):
        at = ApplyToCols(ToDatetime(), cols=s.all(), allow_reject=False)
        X = df_module.make_dataframe(
            {"date": ["2020-01-01", "2020-01-02"], "value": [1, 2]}
        )
        at.fit(X)
