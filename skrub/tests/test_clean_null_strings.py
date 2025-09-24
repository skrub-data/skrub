import numpy as np
import pytest

from skrub._apply_to_cols import RejectColumn
from skrub._clean_null_strings import CleanNullStrings, _trim_whitespace_only


def test_clean_null_strings(df_module):
    s = df_module.make_column("c", ["a", "b", "   ", "N/A", None])
    out = CleanNullStrings().fit_transform(s)
    expected = df_module.make_column("c", ["a", "b", None, None, None])
    df_module.assert_column_equal(out, expected)


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
