import numpy as np
import pytest

from skrub._clean_null_strings import CleanNullStrings, _trim_whitespace_only
from skrub._single_column_transformer import RejectColumn


def test_clean_null_strings(df_module):
    s = df_module.make_column("c", ["a", "b", "   ", "N/A", "None", "none", None])
    out = CleanNullStrings().fit_transform(s)
    expected = df_module.make_column("c", ["a", "b", None, None, None, None, None])
    df_module.assert_column_equal(out, expected)


def test_custom_null_strings(df_module):
    s = df_module.make_column("c", ["a", "b", "   ", "N/A", "foo", None])
    out = CleanNullStrings(null_strings=["foo"]).fit_transform(s)
    expected = df_module.make_column("c", ["a", "b", None, None, None, None])
    df_module.assert_column_equal(out, expected)


def test_custom_null_strings_type(df_module):
    s = df_module.make_column("c", ["a", "b", "   ", "N/A", "foo", None])
    with pytest.raises(ValueError, match=".*a sequence of strictly strings."):
        CleanNullStrings(null_strings=0).fit_transform(s)


def test_reject_non_string_cols(df_module):
    with pytest.raises(RejectColumn, match=".*does not contain strings"):
        CleanNullStrings().fit(df_module.example_column)
    s = df_module.make_column("c", ["a", "b", "   ", "N/A", None])
    cleaner = CleanNullStrings().fit(s)
    # at transform time columns are not rejected anymore
    out = cleaner.transform(df_module.example_column)
    # in this case no substitutions need to be made
    assert out is df_module.example_column


def test_error_trim_whitespace_only():
    # Make codecov happy
    with pytest.raises(TypeError, match="Expecting a Pandas or Polars Series"):
        _trim_whitespace_only(np.array([1]))
