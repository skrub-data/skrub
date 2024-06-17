import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_array_equal
from sklearn.feature_extraction.text import HashingVectorizer

from skrub import fuzzy_join
from skrub._dataframe import _common as ns
from skrub._dataframe._testing_utils import assert_frame_equal


@pytest.mark.parametrize(
    "analyzer",
    ["char", "char_wb", "word"],
)
def test_fuzzy_join(df_module, analyzer):
    """
    Testing if ``fuzzy_join`` results are as expected.
    """
    if df_module.name == "polars":
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'.")
    df1 = df_module.make_dataframe({"a1": ["ana", "lala", "nana et sana", np.nan]})
    df2 = df_module.make_dataframe(
        {"a2": ["anna", "lala et nana", "lana", "sana", np.nan]}
    )

    df_joined = fuzzy_join(
        left=df1,
        right=df2,
        left_on="a1",
        right_on="a2",
        max_dist=1.0,
        add_match_info=True,
    )

    n_cols = ns.shape(df1)[1] + ns.shape(df2)[1] + 3

    assert ns.shape(df_joined) == (len(df1), n_cols)

    df_joined2 = fuzzy_join(
        df2,
        df1,
        left_on="a2",
        right_on="a1",
        max_dist=1.0,
        add_match_info=True,
    )
    # Joining is always done on the left table and thus takes it shape:
    assert ns.shape(df_joined2) == (len(df2), n_cols)

    # TODO: dispatch ``with_columns``
    df1["a2"] = 1

    df_on = fuzzy_join(df_joined, df1, on="a1", suffix="2")
    assert "a12" in ns.column_names(df_on)


def test_max_dist(df_module):
    if df_module.name == "polars":
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'.")
    left = df_module.make_dataframe({"A": ["aa", "bb"]})
    right = df_module.make_dataframe({"A": ["aa", "ba"], "B": [1, 2]})
    join = fuzzy_join(left, right, on="A", suffix="r")
    assert ns.to_list(ns.col(join, "Br")) == [1, 2]
    join = fuzzy_join(left, right, on="A", suffix="r", max_dist=0.5)
    assert ns.to_list(ns.fill_nulls(ns.col(join, "Br"), -1)) == [1, -1]


def test_perfect_matches(df_module):
    if df_module.name == "polars":
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'.")
    # non-regression test for https://github.com/skrub-data/skrub/issues/764
    # fuzzy_join when all rows had a perfect match used to trigger a division by 0
    df = df_module.make_dataframe({"A": [0, 1]})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.filterwarnings("ignore", message="This feature is still experimental")
        join = fuzzy_join(df, df, on="A", suffix="r", add_match_info=True)
    assert_array_equal(
        ns.to_numpy(ns.col(join, "skrub_Joiner_rescaled_distance")), [0.0, 0.0]
    )


def test_fuzzy_join_dtypes(df_module):
    """
    Test that the dtypes of dataframes are maintained after join.
    """
    if df_module.name == "polars":
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'.")
    a = df_module.make_dataframe({"col1": ["aaa", "bbb"], "col2": [1, 2]})
    b = df_module.make_dataframe({"col1": ["aaa_", "bbb_"], "col3": [1, 2]})
    c = fuzzy_join(a, b, on="col1", suffix="r")
    assert ns.dtype(ns.col(a, "col2")).kind == "i"
    assert ns.dtype(ns.col(c, "col2")) == ns.dtype(ns.col(a, "col2"))
    assert ns.dtype(ns.col(c, "col3r")) == ns.dtype(ns.col(b, "col3"))


def test_missing_keys(df_module):
    if df_module.name == "polars":
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'.")
    a = df_module.make_dataframe({"col1": ["aaa", "bbb"], "col2": [1, 2]})
    b = df_module.make_dataframe({"col1": ["aaa_", "bbb_"], "col3": [1, 2]})
    with pytest.raises(
        ValueError,
        match=r"Must pass",
    ):
        fuzzy_join(a, b, left_on="col1", suffix="r", add_match_info=False)
    left = df_module.make_dataframe({"a": ["aa", np.nan, "bb"], "b": [1, 2, np.nan]})
    right = df_module.make_dataframe(
        {"a": ["aa", "bb", np.nan, "cc", "dd"], "c": [5, 6, 7, 8, np.nan]}
    )
    output = fuzzy_join(left, right, on="a", suffix="r", add_match_info=False)
    assert ns.shape(output) == (3, 4)


def test_drop_unmatched(df_module):
    if df_module.name == "polars":
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'.")
    a = df_module.make_dataframe({"col1": ["aaaa", "bbb", "ddd dd"], "col2": [1, 2, 3]})
    b = df_module.make_dataframe(
        {"col1": ["aaa_", "bbb_", "cc ccc"], "col3": [1, 2, 3]}
    )

    c1 = fuzzy_join(
        a,
        b,
        on="col1",
        max_dist=0.9,
        drop_unmatched=True,
        suffix="r",
        add_match_info=False,
    )
    assert ns.shape(c1) == (2, 4)
    c2 = fuzzy_join(a, b, on="col1", max_dist=0.9, suffix="r", add_match_info=False)
    assert sum(ns.is_null(ns.col(c2, "col3r"))) > 0


def test_fuzzy_join_pandas_comparison():
    """
    Tests if fuzzy_join's output is as similar as
    possible with `pandas.merge`.
    """
    left = pd.DataFrame(
        {
            "key": ["K0", "K1", "K2", "K3"],
            "A": ["A0", "A1", "A2", "A3"],
            "B": ["B0", "B1", "B2", "B3"],
        }
    )

    right = pd.DataFrame(
        {
            "key_": ["K0", "K1", "K2", "K3"],
            "C": ["C0", "C1", "C2", "C3"],
            "D": ["D0", "D1", "D2", "D3"],
        }
    )

    result = pd.merge(left, right, left_on="key", right_on="key_")
    result_fj = fuzzy_join(
        left, right, left_on="key", right_on="key_", add_match_info=False
    )

    assert_frame_equal(result, result_fj)


def test_correct_encoder(df_module):
    """
    Test that the encoder error checking is working as intended.
    """
    if df_module.name == "polars":
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'.")

    class TestVectorizer(HashingVectorizer):
        """
        Implements a custom vectorizer to check if the `encoder`
        parameter uses the passed instance as expected.
        Raises an error when `fit` is called.
        """

        def fit(self, X, y=None):
            raise AssertionError("Custom vectorizer was called as intended.")

    left = df_module.make_dataframe(
        {
            "key": ["K0", "K1", "K2", "K3"],
            "A": ["A0", "A1", "A2", "A3"],
            "B": ["B0", "B1", "B2", "B3"],
        }
    )

    right = df_module.make_dataframe(
        {
            "key": ["K0", "K1", "K2", "K3"],
            "C": ["C0", "C1", "C2", "C3"],
            "D": ["D0", "D1", "D2", "D3"],
        }
    )

    enc = TestVectorizer()

    with pytest.raises(
        AssertionError, match=r"Custom vectorizer was called as intended."
    ):
        fuzzy_join(
            left, right, on="key", suffix="_", string_encoder=enc, add_match_info=False
        )


def test_numerical_column(df_module):
    """
    Testing that ``fuzzy_join`` works with numerical columns.
    """
    if df_module.name == "polars":
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'.")
    left = df_module.make_dataframe({"str1": ["aa", "a", "bb"], "int": [10, 2, 5]})
    right = df_module.make_dataframe(
        {
            "str2": ["aa", "bb", "a", "cc", "dd"],
            "int": [55, 6, 2, 15, 6],
        }
    )

    fj_num = fuzzy_join(left, right, on="int", suffix="r", add_match_info=False)
    n_cols = ns.shape(left)[1] + ns.shape(right)[1]
    n_samples = len(left)

    assert ns.shape(fj_num) == (n_samples, n_cols)

    fj_num2 = fuzzy_join(left, right, on="int", add_match_info=True, suffix="r")
    assert ns.shape(fj_num2) == (n_samples, n_cols + 3)

    fj_num3 = fuzzy_join(
        left,
        right,
        on="int",
        max_dist=3,
        drop_unmatched=True,
        suffix="r",
        ref_dist="no_rescaling",
        add_match_info=False,
    )
    assert ns.shape(fj_num3) == (2, n_cols)


def test_datetime_column(df_module):
    """
    Testing that ``fuzzy_join`` works with datetime columns.
    """
    if df_module.name == "polars":
        pytest.xfail(reason="Module 'polars' has no attribute 'to_datetime'.")
    left = df_module.make_dataframe(
        {
            "str1": ["aa", "a", "bb"],
            "date": df_module.module.to_datetime(
                ["10/10/2022", "12/11/2021", "09/25/2011"]
            ),
        }
    )
    right = df_module.make_dataframe(
        {
            "str2": ["aa", "bb", "a", "cc", "dd"],
            "date": df_module.module.to_datetime(
                ["09/10/2022", "12/24/2021", "09/25/2010", "11/05/2025", "02/21/2000"]
            ),
        }
    )

    fj_time = fuzzy_join(left, right, on="date", suffix="r", add_match_info=False)

    fj_time_expected = df_module.make_dataframe(
        {
            "str1": ["aa", "a", "bb"],
            "date": df_module.module.to_datetime(
                ["10/10/2022", "12/11/2021", "09/25/2011"]
            ),
            "str2r": ["aa", "bb", "a"],
            "dater": df_module.module.to_datetime(
                ["09/10/2022", "12/24/2021", "09/25/2010"]
            ),
        }
    )
    assert_frame_equal(fj_time, fj_time_expected)

    n_cols = ns.shape(left)[1] + ns.shape(right)[1]
    n_samples = len(left)

    assert ns.shape(fj_time) == (n_samples, n_cols)

    fj_time2 = fuzzy_join(left, right, on="date", add_match_info=True, suffix="r")
    assert ns.shape(fj_time2) == (n_samples, n_cols + 3)

    fj_time3 = fuzzy_join(
        left,
        right,
        on="date",
        max_dist=0.1,
        drop_unmatched=True,
        suffix="r",
        add_match_info=False,
    )
    assert ns.shape(fj_time3) == (2, n_cols)


def test_mixed_joins(df_module):
    """
    Test fuzzy joining on mixed and multiple column types.
    """
    if df_module.name == "polars":
        pytest.xfail(reason="Module 'polars' has no attribute 'to_datetime'")
    left = df_module.make_dataframe(
        {
            "str1": ["Paris", "Paris", "Paris"],
            "str2": ["Texas", "France", "Greek God"],
            "int1": [10, 2, 5],
            "int2": [103, 250, 532],
            "date": df_module.module.to_datetime(
                ["10/10/2022", "12/11/2021", "09/25/2011"]
            ),
        }
    )
    right = df_module.make_dataframe(
        {
            "str_1": ["Paris", "Paris", "Paris", "cc", "dd"],
            "str_2": ["TX", "FR", "GR Mytho", "cc", "dd"],
            "int1": [55, 6, 2, 15, 6],
            "int2": [554, 146, 32, 215, 612],
            "date": df_module.module.to_datetime(
                ["09/10/2022", "12/24/2021", "09/25/2010", "11/05/2025", "02/21/2000"]
            ),
        }
    )

    # On multiple numeric keys
    fj_num = fuzzy_join(
        left, right, on=["int1", "int2"], suffix="r", add_match_info=False
    )

    expected_fj_num = df_module.make_dataframe(
        {
            "str1": ["Paris", "Paris", "Paris"],
            "str2": ["Texas", "France", "Greek God"],
            "int1": [10, 2, 5],
            "int2": [103, 250, 532],
            "date": df_module.module.to_datetime(
                ["10/10/2022", "12/11/2021", "09/25/2011"]
            ),
            "str_1r": ["Paris", "Paris", "dd"],
            "str_2r": ["FR", "FR", "dd"],
            "int1r": [6, 6, 6],
            "int2r": [146, 146, 612],
            "dater": df_module.module.to_datetime(
                ["12/24/2021", "12/24/2021", "02/21/2000"]
            ),
        }
    )
    assert_frame_equal(fj_num, expected_fj_num)
    assert ns.shape(fj_num) == (3, 10)

    # On multiple string keys
    fj_str = fuzzy_join(
        left,
        right,
        left_on=["str1", "str2"],
        right_on=["str_1", "str_2"],
        suffix="r",
        add_match_info=False,
    )

    expected_fj_str = df_module.make_dataframe(
        {
            "str1": ["Paris", "Paris", "Paris"],
            "str2": ["Texas", "France", "Greek God"],
            "int1": [10, 2, 5],
            "int2": [103, 250, 532],
            "date": df_module.module.to_datetime(
                ["2022-10-10", "2021-12-11", "2011-09-25"]
            ),
            "str_1r": ["Paris", "Paris", "Paris"],
            "str_2r": ["TX", "FR", "GR Mytho"],
            "int1r": [55, 6, 2],
            "int2r": [554, 146, 32],
            "dater": df_module.module.to_datetime(
                ["2022-09-10", "2021-12-24", "2010-09-25"]
            ),
        }
    )
    assert_frame_equal(fj_str, expected_fj_str)
    assert ns.shape(fj_str) == (3, 10)

    # On mixed, numeric and string keys
    fj_mixed = fuzzy_join(
        left,
        right,
        left_on=["str1", "str2", "int2"],
        right_on=["str_1", "str_2", "int2"],
        suffix="r",
        add_match_info=False,
    )
    expected_fj_mixed = df_module.make_dataframe(
        {
            "str1": ["Paris", "Paris", "Paris"],
            "str2": ["Texas", "France", "Greek God"],
            "int1": [10, 2, 5],
            "int2": [103, 250, 532],
            "date": df_module.module.to_datetime(
                ["2022-10-10", "2021-12-11", "2011-09-25"]
            ),
            "str_1r": ["Paris", "Paris", "Paris"],
            "str_2r": ["FR", "FR", "TX"],
            "int1r": [6, 6, 55],
            "int2r": [146, 146, 554],
            "dater": df_module.module.to_datetime(
                ["2021-12-24", "2021-12-24", "2022-09-10"]
            ),
        }
    )
    assert_frame_equal(fj_mixed, expected_fj_mixed)
    assert ns.shape(fj_mixed) == (3, 10)

    # On mixed time and string keys
    fj_mixed2 = fuzzy_join(
        left,
        right,
        left_on=["str1", "date"],
        right_on=["str_1", "date"],
        suffix="r",
        add_match_info=False,
    )
    expected_fj_mixed2 = df_module.make_dataframe(
        {
            "str1": ["Paris", "Paris", "Paris"],
            "str2": ["Texas", "France", "Greek God"],
            "int1": [10, 2, 5],
            "int2": [103, 250, 532],
            "date": df_module.module.to_datetime(
                ["2022-10-10", "2021-12-11", "2011-09-25"]
            ),
            "str_1r": ["Paris", "Paris", "Paris"],
            "str_2r": ["TX", "FR", "GR Mytho"],
            "int1r": [55, 6, 2],
            "int2r": [554, 146, 32],
            "dater": df_module.module.to_datetime(
                ["2022-09-10", "2021-12-24", "2010-09-25"]
            ),
        }
    )
    assert_frame_equal(fj_mixed2, expected_fj_mixed2)
    assert ns.shape(fj_mixed2) == (3, 10)

    # On mixed time and numbers keys
    fj_mixed3 = fuzzy_join(
        left,
        right,
        left_on=["int1", "date"],
        right_on=["int1", "date"],
        suffix="r",
        add_match_info=False,
    )
    expected_fj_mixed3 = df_module.make_dataframe(
        {
            "str1": ["Paris", "Paris", "Paris"],
            "str2": ["Texas", "France", "Greek God"],
            "int1": [10, 2, 5],
            "int2": [103, 250, 532],
            "date": df_module.module.to_datetime(
                ["2022-10-10", "2021-12-11", "2011-09-25"]
            ),
            "str_1r": ["Paris", "Paris", "Paris"],
            "str_2r": ["FR", "FR", "GR Mytho"],
            "int1r": [6, 6, 2],
            "int2r": [146, 146, 32],
            "dater": df_module.module.to_datetime(
                ["2021-12-24", "2021-12-24", "2010-09-25"]
            ),
        }
    )
    assert_frame_equal(fj_mixed3, expected_fj_mixed3)
    assert ns.shape(fj_mixed3) == (3, 10)


def test_iterable_input(df_module):
    """
    Test if iterable inputs (list, set, dictionary or tuple) work.
    """
    if df_module.name == "polars":
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'.")
    df1 = df_module.make_dataframe(
        {"a": ["ana", "lala", "nana"], "str2": ["Texas", "France", "Greek God"]}
    )
    df2 = df_module.make_dataframe(
        {
            "a": ["anna", "lala", "ana", "nnana"],
            "str_2": ["TX", "FR", "GR Mytho", "dd"],
        }
    )
    assert ns.shape(
        fuzzy_join(df1, df2, on=["a"], suffix="r", add_match_info=False)
    ) == (
        3,
        4,
    )
    assert ns.shape(fuzzy_join(df1, df2, on="a", suffix="r", add_match_info=False)) == (
        3,
        4,
    )

    assert ns.shape(
        fuzzy_join(
            df1,
            df2,
            left_on=["a", "str2"],
            right_on=("a", "str_2"),
            suffix="r",
            add_match_info=False,
        )
    ) == (3, 4)
    assert ns.shape(
        fuzzy_join(
            df1,
            df2,
            left_on=("a", "str2"),
            right_on=np.asarray(("a", "str_2")),
            suffix="r",
            add_match_info=False,
        )
    ) == (3, 4)


@pytest.mark.xfail
def test_missing_values(df_module):
    """
    Test fuzzy joining on missing values.
    """
    if df_module.name == "polars":
        pytest.xfail(reason="Polars DataFrame object has no attribute 'reset_index'.")
    a = df_module.make_dataframe({"col1": ["aaaa", "bbb", "ddd dd"], "col2": [1, 2, 3]})
    b = df_module.make_dataframe({"col3": [np.nan, "bbb", "ddd dd"], "col4": [1, 2, 3]})

    with pytest.warns(UserWarning, match=r"merging on missing values"):
        c = fuzzy_join(a, b, left_on="col1", right_on="col3", add_match_info=False)
    assert ns.shape(c)[0] == len(b)

    with pytest.warns(UserWarning, match=r"merging on missing values"):
        c = fuzzy_join(b, a, left_on="col3", right_on="col1", add_match_info=True)
    assert ns.shape(c)[0] == len(b)
