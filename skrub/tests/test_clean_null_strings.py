import pytest

from skrub._clean_null_strings import CleanNullStrings
from skrub._on_each_column import RejectColumn


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
