import datetime

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OrdinalEncoder

from skrub import ApplyToCols
from skrub import _dataframe as sbd
from skrub import selectors as s
from skrub._to_datetime import ToDatetime
from skrub.core import RejectColumn


def test_single_column_transformer_becomes_apply_to_each_col(df_module):
    """SingleColumnTransformer should be wrapped as ApplyToEachCol."""
    at = ApplyToCols(ToDatetime(), cols=s.all())
    X = df_module.make_dataframe({"date_col": ["2020-01-01", "2020-01-02"]})
    at.fit(X)
    assert hasattr(at, "transformers_")
    assert isinstance(at.transformers_, dict)


def test_non_single_column_transformer_becomes_apply_to_subframe(df_module):
    """Non-SingleColumnTransformer should be wrapped as ApplyToSubFrame."""
    at = ApplyToCols(OrdinalEncoder(), cols=s.all())
    X = df_module.make_dataframe({"col1": ["a", "b"], "col2": ["x", "y"]})
    at.fit(X)
    assert hasattr(at, "transformer_")


def test_invalid_parameters():
    """all these parameters should be boolean."""

    X = None  # Placeholder for the dataframe, not used in this test

    with pytest.raises((TypeError, RuntimeError), match=r"allow_reject.*bool"):
        at = ApplyToCols(ToDatetime(), allow_reject="yes")
        at.fit_transform(X)
    with pytest.raises((TypeError, RuntimeError), match=r"keep_original.*bool"):
        at = ApplyToCols(ToDatetime(), keep_original="no")
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

    with pytest.raises((RejectColumn, RuntimeError)):
        at = ApplyToCols(ToDatetime(), cols=s.all(), allow_reject=False)
        X = df_module.make_dataframe(
            {"date": ["2020-01-01", "2020-01-02"], "value": [1, 2]}
        )
        at.fit(X)


@pytest.mark.parametrize(
    "transformer,data",
    [
        (ToDatetime(), {"date_col": ["2020-01-01", "2020-01-02"]}),
        (OrdinalEncoder(), {"col1": ["a", "b"], "col2": ["x", "y"]}),
    ],
)
def test_check_is_fitted_transform(df_module, transformer, data):
    """Test that transform raises NotFittedError before fitting."""
    at = ApplyToCols(transformer, cols=s.all())
    X = df_module.make_dataframe(data)

    # Should raise NotFittedError when calling transform before fit
    with pytest.raises(NotFittedError):
        at.transform(X)


def test_check_is_fitted_get_feature_names_out():
    """Test that get_feature_names_out raises NotFittedError before fitting."""
    at = ApplyToCols(ToDatetime(), cols=s.all())

    # Should raise NotFittedError when calling get_feature_names_out before fit
    with pytest.raises(NotFittedError):
        at.get_feature_names_out()


def test_get_feature_names_out_after_fit(df_module):
    """Test that get_feature_names_out works after fitting."""
    at = ApplyToCols(ToDatetime(), cols=s.all())
    X = df_module.make_dataframe({"date_col": ["2020-01-01", "2020-01-02"]})
    at.fit(X)

    feature_names = at.get_feature_names_out()
    assert feature_names == ["date_col"]


def test_getattr_raises_for_wrong_attribute(df_module):
    """Test __getattr__ raises proper AttributeError for wrong attributes."""
    # Test that accessing transformers_ on non-single-column transformer raises error
    at = ApplyToCols(OrdinalEncoder())
    X = df_module.make_dataframe({"col1": ["a", "b"], "col2": ["x", "y"]})

    at.fit(X)

    with pytest.raises(
        AttributeError,
        match="'transformers_' is only available for single-column transformers",
    ):
        _ = at.transformers_

    delattr(at, "transformer_")

    # artificially remove transformer_ to test that accessing it raises error
    with pytest.raises(
        AttributeError, match="ApplyToCols.*has no attribute.*transformer_"
    ):
        _ = at.transformer_

    # Test that accessing transformer_ on single-column transformer raises error
    at = ApplyToCols(ToDatetime())
    X = df_module.make_dataframe({"date_col": ["2020-01-01", "2020-01-02"]})
    at.fit(X)

    with pytest.raises(
        AttributeError,
        match="'transformer_' is only available for non-single-column transformers",
    ):
        _ = at.transformer_

    delattr(at, "transformers_")

    # artificially remove transformers_ to test that accessing it raises error
    with pytest.raises(
        AttributeError, match="ApplyToCols.*has no attribute.*transformers_"
    ):
        _ = at.transformers_

    # Test that accessing any non-existent attribute raises error
    with pytest.raises(AttributeError, match="ApplyToCols.*has no attribute.*foo"):
        _ = at.foo
