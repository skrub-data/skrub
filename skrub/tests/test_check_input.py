import numpy as np
import pandas as pd
import pytest

from skrub import _dataframe as sbd
from skrub._check_input import CheckInputDataFrame


def test_good_input(df_module):
    check = CheckInputDataFrame()
    df = df_module.example_dataframe
    assert check.fit_transform(df) is df
    assert check.transform(df) is df
    assert check.module_name_ == df_module.name
    assert check.get_feature_names_out() == sbd.column_names(df)


def test_input_is_an_array():
    a = np.ones((3, 2))
    check = CheckInputDataFrame()

    # 2D array in fit
    with pytest.warns(UserWarning, match="Only .* DataFrames are supported"):
        d = check.fit_transform(a)
    assert sbd.is_pandas(d)
    assert sbd.is_dataframe(d)
    assert sbd.column_names(d) == ["0", "1"]

    # 2D array in transform
    with pytest.warns(UserWarning, match="Only .* DataFrames are supported"):
        d = check.transform(a)
    assert sbd.is_pandas(d)
    assert sbd.is_dataframe(d)
    assert sbd.column_names(d) == ["0", "1"]

    # 1D array
    with pytest.raises(ValueError, match=".*incompatible shape"):
        check.fit_transform(np.ones((2,)))


def test_wrong_dataframe_library_in_transform():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [0, 1], "b": [10, 20]})
    check = CheckInputDataFrame().fit(df)
    with pytest.raises(
        TypeError,
        match=".* fitted to a polars dataframe .* applied to a pandas dataframe",
    ):
        check.transform(df.to_pandas())


def test_column_names_to_unique_strings():
    df = pd.DataFrame(np.ones((2, 4)), columns=["a", 0, "0", "a"])
    assert df.columns.tolist() == ["a", 0, "0", "a"]
    check = CheckInputDataFrame()
    with pytest.warns(UserWarning, match="Some column names are not strings"):
        with pytest.warns(UserWarning, match="Found duplicated column names"):
            out = check.fit_transform(df)
    assert out.shape == (2, 4)
    out_cols = out.columns.tolist()
    assert out_cols[:2] == ["a", "0"]
    assert out_cols[2].startswith("0__skrub_")
    assert out_cols[3].startswith("a__skrub_")
    transform_out = check.transform(df)
    assert transform_out.columns.tolist() == out_cols


def test_input_is_sparse():
    df = pd.DataFrame({"a": [1, 2], "b": pd.Series(pd.arrays.SparseArray([0, 1]))})
    with pytest.raises(TypeError, match=".*are sparse Pandas series"):
        CheckInputDataFrame().fit_transform(df)
    dense_df = pd.DataFrame({"a": [1, 2], "b": [0, 1]})
    check = CheckInputDataFrame().fit(dense_df)
    with pytest.raises(TypeError, match=".*are sparse Pandas series"):
        check.transform(df)


def test_input_is_lazy():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1, 2]})
    lazy_df = df.lazy()
    check = CheckInputDataFrame()
    with pytest.warns(UserWarning, match=".*only works on eager"):
        out = check.fit_transform(lazy_df)
    assert isinstance(out, pl.DataFrame)
    from polars.testing import assert_frame_equal

    assert_frame_equal(out, df)

    with pytest.warns(UserWarning, match=".*only works on eager"):
        out = check.transform(lazy_df)
    assert isinstance(out, pl.DataFrame)
    assert_frame_equal(out, df)


def test_column_names_changed(df_module):
    check = CheckInputDataFrame().fit(df_module.example_dataframe)
    new_df = sbd.set_column_names(
        df_module.example_dataframe,
        [f"{c}_changed" for c in sbd.column_names(df_module.example_dataframe)],
    )
    with pytest.raises(ValueError, match="Columns .* differ"):
        check.transform(new_df)


def test_column_names_diff_display():
    """
    >>> import pandas as pd
    >>> from skrub._check_input import CheckInputDataFrame
    >>> df = pd.DataFrame({'a': [1], 'b': [1], 'c': [2]})
    >>> check = CheckInputDataFrame().fit(df)
    >>> df1 = pd.DataFrame({'a': [1], 'b_changed': [1]})
    >>> check.transform(df1)
    Traceback (most recent call last):
        ...
    ValueError: Columns of dataframes passed to fit() and transform() differ:
    a
    + b_changed
    - b
    - c
    """
