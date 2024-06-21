import pandas as pd
import pytest

from skrub import _dataframe as sbd
from skrub._multi_agg_joiner import MultiAggJoiner
from skrub.conftest import _POLARS_INSTALLED


@pytest.fixture
def main_table():
    df = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
        }
    )
    return df


@pytest.mark.parametrize("use_X_placeholder", [False, True])
def test_simple_fit_transform(df_module, main_table, use_X_placeholder):
    "Check the general behaviour of the `MultiAggJoiner`."
    main_table = df_module.DataFrame(main_table)
    aux = [main_table, main_table] if not use_X_placeholder else ["X", "X"]

    multi_agg_joiner = MultiAggJoiner(
        aux_tables=aux,
        main_keys=[["userId"], ["movieId"]],
        aux_keys=[["userId"], ["movieId"]],
        cols=[["rating", "genre"], ["rating"]],
        suffixes=["_user", "_movie"],
    )

    main_user_movie = multi_agg_joiner.fit_transform(main_table)

    expected = df_module.make_dataframe(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
            "genre_mode_user": ["drama", "drama", "drama", "sf", "sf", "sf"],
            "rating_mean_user": [4.0, 4.0, 4.0, 3.0, 3.0, 3.0],
            "rating_mean_movie": [4.0, 4.0, 3.0, 3.0, 3.0, 4.0],
        }
    )
    df_module.assert_frame_equal(
        sbd.pandas_convert_dtypes(main_user_movie), sbd.pandas_convert_dtypes(expected)
    )


def test_X_placeholder(df_module, main_table):
    """
    Check that the 'X' placeholder replaces any of the `aux_tables` into the dataframe
    seen in `fit`.
    """
    main_table = df_module.DataFrame(main_table)

    multi_agg_joiner = MultiAggJoiner(
        aux_tables=["X", main_table],
        keys=[["userId"], ["userId"]],
    )
    multi_agg_joiner.fit_transform(main_table)
    assert multi_agg_joiner._aux_tables == [main_table, main_table]

    multi_agg_joiner = MultiAggJoiner(
        aux_tables=["X", "X"],
        keys=[["userId"], ["userId"]],
    )
    multi_agg_joiner.fit_transform(main_table)
    assert multi_agg_joiner._aux_tables == [main_table, main_table]

    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table, main_table, "X", main_table],
        keys=[["userId"], ["userId"], ["userId"], ["userId"]],
    )
    multi_agg_joiner.fit_transform(main_table)
    assert multi_agg_joiner._aux_tables == [
        main_table,
        main_table,
        main_table,
        main_table,
    ]


def test_wrong_aux_tables(df_module, main_table):
    "Check that wrong `aux_tables` parameters for the `MultiAggJoiner` raise an error."
    main_table = df_module.DataFrame(main_table)

    # Check aux_tables isn't an array
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=main_table,
        keys=["userId"],
    )
    with pytest.raises(
        ValueError,
        match=r"(?=must be an iterable containing dataframes and/or the string 'X')",
    ):
        multi_agg_joiner.fit_transform(main_table)

    # Check aux_tables is not a dataframe or "X"
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[1],
        keys=["userId"],
    )
    with pytest.raises(
        ValueError,
        match=r"(?=must be an iterable containing dataframes and/or the string 'X')",
    ):
        multi_agg_joiner.fit_transform(main_table)


def test_wrong_main_table(df_module, main_table):
    "Check that wrong `X` parameters in the `MultiAggJoiner` `fit` raise an error."
    main_table = df_module.DataFrame(main_table)

    # Check wrong `X`
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
        keys=[["userId"]],
    )
    with pytest.raises(
        TypeError, match=r"`X` must be a dataframe, got <class 'list'>."
    ):
        multi_agg_joiner.fit_transform([1])


@pytest.mark.skipif(not _POLARS_INSTALLED, reason="Polars not available.")
def test_check_wrong_aux_table_type(main_table, df_module):
    """
    Check that providing different types for `X` and `aux_tables`
    in the `MultiAggJoiner` raises an error.
    """
    import polars as pl

    other_px = pd if df_module.module is pl else pl
    main_table = df_module.DataFrame(main_table)
    aux_table = other_px.DataFrame(main_table)

    # Check aux_tables is pandas when X is polars or the opposite
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[aux_table],
        keys=[["userId"]],
    )
    wanted_type = "Pandas" if df_module.module == pd else "Polars"
    with pytest.raises(
        TypeError, match=rf"All `aux_tables` must be {wanted_type} dataframes."
    ):
        multi_agg_joiner.fit_transform(main_table)


def test_correct_keys(main_table, df_module):
    "Check that expected `keys` parameters for the `MultiAggJoiner` are working."
    main_table = df_module.DataFrame(main_table)

    # Check only keys
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
        keys=[["userId"]],
    )
    multi_agg_joiner.fit_transform(main_table)
    assert multi_agg_joiner._main_keys == [["userId"]]
    assert multi_agg_joiner._aux_keys == [["userId"]]

    # Check multiple keys
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
        keys=[["userId", "movieId"]],
    )
    multi_agg_joiner.fit_transform(main_table)

    # Check keys multiple tables
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table, main_table],
        keys=[["userId"], ["userId"]],
    )
    multi_agg_joiner.fit_transform(main_table)

    # Check multiple main_keys and aux_keys, same length
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
        main_keys=[["userId", "movieId"]],
        aux_keys=[["userId", "movieId"]],
    )
    multi_agg_joiner.fit_transform(main_table)


def test_no_keys(main_table, df_module):
    "Check that no `keys` for the `MultiAggJoiner` raise an error."
    main_table = df_module.DataFrame(main_table)

    # Check no keys at all
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
    )
    error_msg = r"Must pass either `keys`, or \(`main_keys` and `aux_keys`\)."
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main_table)


def test_too_many_keys(main_table, df_module):
    "Check that providing too many keys for the `MultiAggJoiner` raise an error."
    main_table = df_module.DataFrame(main_table)

    # Check too many main_keys
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
        main_keys=[["userId", "movieId"]],
        aux_keys=[["userId"]],
    )
    with pytest.raises(
        ValueError,
        match=(
            r"(?=.*`main_keys` and `aux_keys` elements have different lengths at"
            r" position 0)"
        ),
    ):
        multi_agg_joiner.fit_transform(main_table)

    # Check too many aux_keys
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
        main_keys=[["userId"]],
        aux_keys=[["userId", "movieId"]],
    )
    with pytest.raises(
        ValueError, match=r"(?=.*Cannot join on different numbers of columns)"
    ):
        multi_agg_joiner.fit_transform(main_table)

    # Check providing keys and extra main_keys
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
        keys=[["userId"]],
        main_keys=[["userId"]],
    )
    with pytest.raises(ValueError, match=r"(?=.*not a combination of both.)"):
        multi_agg_joiner.fit_transform(main_table)

    # Check providing keys and extra aux_keys
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
        keys=[["userId"]],
        aux_keys=[["userId"]],
    )
    with pytest.raises(ValueError, match=r"(?=.*not a combination of both.)"):
        multi_agg_joiner.fit_transform(main_table)


def test_unknown_keys(main_table, df_module):
    "Check that providing unknown keys in the `MultiAggJoiner` raise an error."
    main_table = df_module.DataFrame(main_table)

    # Check main_keys doesn't exist in table
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
        main_keys=[["wrong_key"]],
        aux_keys=[["userId"]],
    )
    error_msg = r"(?=.*columns cannot be used because they do not exist)"
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main_table)

    # Check aux_keys doesn't exist in table
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
        main_keys=[["userId"]],
        aux_keys=[["wrong_key"]],
    )
    error_msg = r"(?=.*columns cannot be used because they do not exist)"
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main_table)


def test_wrong_keys_length(main_table, df_module):
    "Check that providing wrong key lengths in the `MultiAggJoiner` raise an error."
    main_table = df_module.DataFrame(main_table)

    # Check wrong main_keys lenght
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table, main_table],
        main_keys=[["userId"]],
        aux_keys=[["userId"], ["userId"]],
    )
    error_msg = r"(?=The length of `main_keys` must match the number of `aux_tables`)"
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main_table)

    # Check wrong aux_keys length
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table, main_table],
        main_keys=[["userId"], ["userId"]],
        aux_keys=[["userId"]],
    )
    error_msg = r"(?=The length of `aux_keys` must match the number of `aux_tables`)"
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main_table)


def test_default_cols(main_table, df_module):
    """
    Check that by default, `cols` is set to a list of list. For each table in
    `aux_tables`, the corresponding list will be all columns of that table,
    except the `aux_keys` associated with that table.
    """
    main_table = df_module.DataFrame(main_table)

    # Check no cols
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table, main_table],
        keys=[["userId", "movieId"], ["userId"]],
    )
    multi_agg_joiner.fit(main_table)
    multi_agg_joiner._cols == [["rating", "genre"], ["movieId", "rating", "genre"]]


def test_correct_cols(main_table, df_module):
    "Check that expected `cols` parameters for the `MultiAggJoiner` are working."
    main_table = df_module.DataFrame(main_table)

    # Check providing one col
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
        keys=[["userId"]],
        cols=[["rating"]],
    )
    multi_agg_joiner.fit_transform(main_table)
    assert multi_agg_joiner._cols == [["rating"]]

    # Check providing one col for each aux_tables
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table, main_table],
        keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
    )
    multi_agg_joiner.fit_transform(main_table)
    assert multi_agg_joiner._cols == [["rating"], ["rating"]]


def test_wrong_cols_input_type(main_table, df_module):
    """Check that the wrong `cols` type in the `MultiAggJoiner` raises an error."""
    main_table = df_module.DataFrame(main_table)

    # Check providing wrong cols type
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
        keys=[["userId"]],
        cols=[[1]],
    )
    error_msg = r"Accepted inputs for `cols` are None and iterable of iterable of str."
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main_table)


def test_too_many_cols(main_table, df_module):
    """
    Check that providing more `cols` than `aux_tables`
    in the `MultiAggJoiner` raises an error.
    """
    main_table = df_module.DataFrame(main_table)

    # Check providing too many cols
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
        keys=[["userId"]],
        cols=[["rating"], ["rating"]],
    )
    error_msg = (
        r"The number of provided cols must match the number of tables in `aux_tables`."
    )
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main_table)


def test_cols_not_in_table(main_table, df_module):
    """
    Check that providing a `cols` not in `aux_tables`
    in the `MultiAggJoiner` raises an error.
    """
    main_table = df_module.DataFrame(main_table)

    # Check cols not in table
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
        keys=[["userId"]],
        cols=[["wrong_col"]],
    )
    error_msg = r"(?=.*columns cannot be used because they do not exist)"
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main_table)


def test_default_operations(main_table, df_module):
    """
    Check that the default `operations` in the `MultiAggJoiner` is an
    iterable of ['mode', 'mean'].
    """
    main_table = df_module.DataFrame(main_table)

    # Check default operations
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table, main_table],
        keys=[["userId"], ["userId"]],
        cols=[["rating", "genre"], ["rating", "genre"]],
    )
    multi_agg_joiner.fit(main_table)
    assert multi_agg_joiner._operations == [["mean", "mode"], ["mean", "mode"]]


def test_correct_operations(main_table, df_module):
    "Check that expected `operations` parameters for the `MultiAggJoiner` are working."
    main_table = df_module.DataFrame(main_table)

    # Check one operation
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
        keys=[["userId"]],
        cols=[["rating"]],
        operations=[["mean"]],
    )
    multi_agg_joiner.fit_transform(main_table)
    assert multi_agg_joiner._operations == [["mean"]]

    # Check one operation for each aux table
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table, main_table],
        keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
        operations=[["mean"], ["mean"]],
    )
    multi_agg_joiner.fit_transform(main_table)
    assert multi_agg_joiner._operations == [["mean"], ["mean"]]

    # Check one and two operations
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table, main_table],
        keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
        operations=[["mean"], ["mean", "mode"]],
    )
    multi_agg_joiner.fit_transform(main_table)
    assert multi_agg_joiner._operations == [["mean"], ["mean", "mode"]]


def test_wrong_operations(main_table, df_module):
    "Check that wrong `operation` parameters for the `MultiAggJoiner` raise an error."
    main_table = df_module.DataFrame(main_table)

    # Check list of operations
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table, main_table],
        keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
        operations="mean",
    )
    error_msg = (
        r"Accepted inputs for `operations` are None and iterable of iterable of str."
    )
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main_table)

    # Check badly formatted operation
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table, main_table],
        keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
        operations=[["mean", "mean", "mode"]],
    )
    error_msg = (
        r"The number of iterables in `operations` must match the number of tables"
    )
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main_table)


def test_default_suffixes(main_table, df_module):
    """
    Check that the default `suffixes` in the `MultiAggJoiner` are the
    table indexes in `aux_tables`.
    """
    main_table = df_module.DataFrame(main_table)

    # check default suffixes with multiple tables
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table, main_table],
        keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
    )
    multi_agg_joiner.fit_transform(main_table)
    assert multi_agg_joiner._suffixes == ["_0", "_1"]


def test_suffixes(main_table, df_module):
    "Check that the `suffixes` parameter of the `MultiAggJoiner` works correctly."
    main_table = df_module.DataFrame(main_table)

    # check suffixes when defined
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table, main_table],
        keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
        suffixes=["_this", "_works"],
    )
    multi_agg_joiner.fit_transform(main_table)
    assert multi_agg_joiner._suffixes == ["_this", "_works"]


def test_too_many_suffixes(main_table, df_module):
    "Check that providing too many `suffixes` to the `MultiAggJoiner` raises an error."
    main_table = df_module.DataFrame(main_table)

    # check too many suffixes
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table, main_table],
        keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
        suffixes=["_0", "_1", "_2"],
    )
    error_msg = (
        r"The number of provided `suffixes` must match the number of tables in"
        r" `aux_tables`. Got 3 suffixes and 2 auxiliary tables."
    )
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main_table)


def test_non_str_suffixes(main_table, df_module):
    "Check that providing non-str `suffixes` to the `MultiAggJoiner` raises an error."
    main_table = df_module.DataFrame(main_table)

    # check suffixes not str
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
        keys=[["userId"]],
        cols=[["rating"]],
        suffixes=[1],
    )
    error_msg = r"Accepted inputs for `suffixes` are None and iterable of str."
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.fit_transform(main_table)


def test_iterable_parameters(main_table, df_module):
    "Check that providing iterable parameters to `MultiAggJoiner` is possible."
    main_table = df_module.DataFrame(main_table)

    # Lists of lists
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table, main_table],
        keys=[["userId"], ["userId"]],
        cols=[["rating"], ["rating"]],
        operations=[["mean"], ["mean", "mode"]],
        suffixes=["_1", "_2"],
    )
    multi_agg_joiner.fit_transform(main_table)

    # Tuples of tuples
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=(main_table, main_table),
        keys=(("userId",), ("userId",)),
        cols=(("rating",), ("rating",)),
        operations=(("mean",), ("mean", "mode")),
        suffixes=("_1", "_2"),
    )
    multi_agg_joiner.fit_transform(main_table)

    # Lists of tuples
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table, main_table],
        keys=[("userId",), ("userId",)],
        cols=[("rating",), ("rating",)],
        operations=[("mean",), ("mean", "mode")],
        suffixes=["_1", "_2"],
    )
    multi_agg_joiner.fit_transform(main_table)

    # Tuples of lists
    multi_agg_joiner = MultiAggJoiner(
        aux_tables=(main_table, main_table),
        keys=(["userId"], ["userId"]),
        cols=(["rating"], ["rating"]),
        operations=(["mean"], ["mean", "mode"]),
        suffixes=("_1", "_2"),
    )
    multi_agg_joiner.fit_transform(main_table)


def test_not_fitted_dataframe(main_table, df_module):
    """
    Check that calling `transform` on a dataframe not containing the columns
    seen during `fit` raises an error.
    """
    main_table = df_module.DataFrame(main_table)
    not_main = df_module.make_dataframe({"wrong": [1, 2, 3], "dataframe": [4, 5, 6]})

    multi_agg_joiner = MultiAggJoiner(
        aux_tables=[main_table],
        keys=[["userId"]],
    )
    multi_agg_joiner.fit(main_table)
    error_msg = r"(?=.*columns cannot be used because they do not exist)"
    with pytest.raises(ValueError, match=error_msg):
        multi_agg_joiner.transform(not_main)
