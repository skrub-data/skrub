import re

import numpy as np
import pandas as pd
import pandas.testing
import pytest
from sklearn.exceptions import NotFittedError

from skrub import _dataframe as sbd
from skrub._agg_joiner import AggJoiner, AggTarget, aggregate, perform_groupby


@pytest.fixture
def main_table(df_module):
    return df_module.make_dataframe(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.1, 4.1, 4.1, 3.1, 2.1, 4.1],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
        }
    )


def test_aggregate_single_operation(df_module, main_table):
    aggregated = aggregate(
        main_table,
        operations=["mean"],
        key=["userId"],
        cols_to_agg=["rating"],
        suffix="",
    )
    # Sorting rows because polars doesn't necessarily maintain order in group_by
    if df_module.name == "polars":
        aggregated = sbd.sort(aggregated, by="userId")
    # In that order because columns are sorted in ``aggregate``
    expected = df_module.make_dataframe({"userId": [1, 2], "rating_mean": [4.1, 3.1]})
    df_module.assert_frame_equal(aggregated, expected)

    aggregated = aggregate(
        main_table,
        operations=["mode"],
        key=["userId"],
        cols_to_agg=["genre"],
        suffix="",
    )
    if df_module.name == "polars":
        aggregated = sbd.sort(aggregated, by="userId")
    expected = df_module.make_dataframe(
        {"userId": [1, 2], "genre_mode": ["drama", "sf"]}
    )
    df_module.assert_frame_equal(aggregated, expected)


def test_aggregate_multiple_operations(df_module, main_table):
    aggregated = aggregate(
        main_table,
        operations=["mean", "sum"],
        key=["userId"],
        cols_to_agg=["rating"],
        suffix="",
    )
    if df_module.name == "polars":
        aggregated = sbd.sort(aggregated, by="userId")
    expected = df_module.make_dataframe(
        {"userId": [1, 2], "rating_mean": [4.1, 3.1], "rating_sum": [12.3, 9.3]}
    )
    df_module.assert_frame_equal(aggregated, expected)

    # Test that the order of the operations is kept in output columns
    aggregated = aggregate(
        main_table,
        operations=["sum", "mean"],
        key=["userId"],
        cols_to_agg=["rating"],
        suffix="",
    )
    if df_module.name == "polars":
        aggregated = sbd.sort(aggregated, by="userId")
    expected = df_module.make_dataframe(
        {"userId": [1, 2], "rating_sum": [12.3, 9.3], "rating_mean": [4.1, 3.1]}
    )
    df_module.assert_frame_equal(aggregated, expected)


def test_aggregate_multiple_columns(df_module):
    main_table = df_module.make_dataframe(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "apples": [1, 2, 3, 4, 5, 6],
            "oranges": [10, 20, 30, 40, 50, 60],
        }
    )

    aggregated = aggregate(
        main_table,
        operations=["sum"],
        key=["userId"],
        cols_to_agg=["apples", "oranges"],
        suffix="",
    )
    if df_module.name == "polars":
        aggregated = sbd.sort(aggregated, by="userId")
    expected = df_module.make_dataframe(
        {"userId": [1, 2], "apples_sum": [6, 15], "oranges_sum": [60, 150]}
    )
    df_module.assert_frame_equal(aggregated, expected)


def test_aggregate_boolean_columns(df_module):
    """Boolean columns are considered non numeric by selectors."""
    main_table = df_module.make_dataframe(
        {"userId": [1, 1, 1, 2, 2, 2], "flag": [False, False, True, False, True, True]}
    )
    aggregated = aggregate(
        main_table,
        operations=["mode"],
        key=["userId"],
        cols_to_agg=["flag"],
        suffix="",
    )
    if df_module.name == "polars":
        aggregated = sbd.sort(aggregated, by="userId")
    expected = df_module.make_dataframe({"userId": [1, 2], "flag_mode": [False, True]})
    df_module.assert_frame_equal(aggregated, expected)


def test_aggregate_suffix(df_module, main_table):
    main_table = df_module.DataFrame(main_table)
    aggregated = aggregate(
        main_table,
        operations=["mean"],
        key=["userId"],
        cols_to_agg=["rating"],
        suffix="_custom_suffix",
    )
    assert sbd.column_names(aggregated) == ["userId", "rating_mean_custom_suffix"]


def test_aggregate_wrong_operation_type(df_module, main_table):
    main_table = df_module.DataFrame(main_table)
    with pytest.raises(
        AttributeError,
        match=(
            r"(removing the following columns: \[\'genre\'\] or the following"
            r" operations: \[\'std\'\])"
        ),
    ):
        aggregate(
            main_table,
            operations=["std"],
            key=["userId"],
            cols_to_agg=["genre"],
            suffix="",
        )


@pytest.mark.parametrize("use_X_placeholder", [False, True])
def test_agg_joiner_simple_fit_transform(df_module, main_table, use_X_placeholder):
    "Check the general behaviour of the `AggJoiner`."
    aux = main_table if not use_X_placeholder else "X"

    agg_joiner_user = AggJoiner(
        aux_table=aux,
        operations="mode",
        aux_key="userId",
        main_key="userId",
        cols="genre",
        suffix="_user",
    )

    main_user = agg_joiner_user.fit_transform(main_table)

    expected_user = df_module.make_dataframe(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.1, 4.1, 4.1, 3.1, 2.1, 4.1],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
            "genre_mode_user": ["drama", "drama", "drama", "sf", "sf", "sf"],
        }
    )
    df_module.assert_frame_equal(main_user, expected_user)

    agg_joiner_movie = AggJoiner(
        aux_table=aux,
        operations="mean",
        aux_key="movieId",
        main_key="movieId",
        cols="rating",
        suffix="_movie",
    )

    main_movie = agg_joiner_movie.fit_transform(main_table)

    expected_movie = df_module.make_dataframe(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.1, 4.1, 4.1, 3.1, 2.1, 4.1],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
            "rating_mean_movie": [4.1, 4.1, 3.1, 3.1, 3.1, 4.1],
        }
    )

    df_module.assert_frame_equal(main_movie, expected_movie)


def test_agg_joiner_wrong_operations(df_module, main_table):
    "Check that a useful error is raised when `operations` is not supported"

    # Test operations is string not in list
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations="wrong_op",
        key="userId",
    )
    with pytest.raises(
        ValueError,
        match=r"(`operations` options are)",
    ):
        agg_joiner.fit(main_table)

    # Test operations is None
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations=None,
        key="userId",
    )
    with pytest.raises(
        ValueError,
        match=r"(`operations` must be a string or an iterable of strings)",
    ):
        agg_joiner.fit(main_table)

    # Test operations is int
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations=2,
        key="userId",
    )
    with pytest.raises(
        ValueError,
        match=r"(`operations` must be a string or an iterable of strings)",
    ):
        agg_joiner.fit(main_table)


def test_agg_joiner_correct_keys(df_module, main_table):
    "Check that expected `key` parameters for the `AggJoiner` are working."

    # Check only key
    agg_joiner = AggJoiner(
        aux_table=main_table, operations="mode", key="userId", cols=["rating", "genre"]
    )
    agg_joiner.fit(main_table)
    assert agg_joiner._main_key == ["userId"]
    assert agg_joiner._aux_key == ["userId"]

    # Check multiple key
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations="mode",
        key=["userId", "movieId"],
        cols=["rating", "genre"],
    )
    agg_joiner.fit(main_table)

    # Check multiple main_key and aux_key, same length
    agg_joiner = AggJoiner(
        aux_table=main_table,
        main_key=["userId", "movieId"],
        aux_key=["userId", "movieId"],
        cols=["rating", "genre"],
        operations="mode",
    )
    agg_joiner.fit(main_table)
    assert agg_joiner._main_key == ["userId", "movieId"]
    assert agg_joiner._aux_key == ["userId", "movieId"]


def test_agg_joiner_wrong_keys(df_module, main_table):
    "Check that wrong `key` parameters for the `AggJoiner` raise an error."

    # Check too many main_key
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations="mode",
        main_key=["userId", "movieId"],
        aux_key="userId",
        cols=["rating", "genre"],
    )
    with pytest.raises(
        ValueError, match=r"(?=.*Cannot join on different numbers of columns)"
    ):
        agg_joiner.fit(main_table)

    # Check too many aux_key
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations="mode",
        main_key="userId",
        aux_key=["userId", "movieId"],
        cols=["rating", "genre"],
    )
    with pytest.raises(
        ValueError, match=r"(?=.*Cannot join on different numbers of columns)"
    ):
        agg_joiner.fit(main_table)

    # Check providing key and extra main_key
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations="mode",
        key="userId",
        main_key="userId",
        cols=["rating", "genre"],
    )
    with pytest.raises(ValueError, match=r"(?=.*not a combination of both.)"):
        agg_joiner.fit(main_table)

    # Check providing key and extra aux_key
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations="mode",
        key="userId",
        aux_key="userId",
        cols=["rating", "genre"],
    )
    with pytest.raises(ValueError, match=r"(?=.*not a combination of both.)"):
        agg_joiner.fit(main_table)

    # Check main_key doesn't exist in table
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations="mode",
        main_key="wrong_key",
        aux_key="userId",
        cols=["rating", "genre"],
    )
    match = r"(?=.*columns cannot be used because they do not exist)"
    with pytest.raises(ValueError, match=match):
        agg_joiner.fit(main_table)

    # Check aux_key doesn't exist in table
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations="mode",
        main_key="userId",
        aux_key="wrong_key",
        cols=["rating", "genre"],
    )
    match = r"(?=.*columns cannot be used because they do not exist)"
    with pytest.raises(ValueError, match=match):
        agg_joiner.fit(main_table)


def test_agg_joiner_default_suffix(df_module, main_table):
    "Check that the default `suffix` of `AggJoiner` is ''."

    # Check no suffix
    agg_joiner = AggJoiner(
        aux_table=main_table, operations="mode", key="userId", cols=["rating", "genre"]
    )
    agg_joiner.fit(main_table)
    assert agg_joiner.suffix == ""


def test_agg_joiner_too_many_suffixes(main_table):
    "Check that providing more than one `suffix` for the `AggJoiner` raises an error."

    # Check inconsistent number of suffixes
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations="mean",
        key="userId",
        cols="rating",
        suffix=["_user", "_movie", "_tag"],
    )
    with pytest.raises(ValueError, match=r"(?='suffix' must be a string.*)"):
        agg_joiner.fit(main_table)


def test_agg_joiner_duplicate_col_name_after_suffix(main_table):
    "Check that ``__skrub_<random string>__`` is added for duplicate column names."

    main_table = sbd.with_columns(
        main_table, **{"rating_mean": [4.1, 4.1, 4.1, 3.1, 2.1, 4.1]}
    )
    # Check inconsistent number of suffixes
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations="mean",
        key="userId",
        cols="rating",
        suffix="",
    )
    aggregated = agg_joiner.fit_transform(main_table)
    assert aggregated.shape == (6, 6)
    assert re.match(r"rating_mean__skrub_[0-9a-f]+__", sbd.column_names(aggregated)[5])


def test_agg_joiner_default_cols(main_table):
    "Check that by default, `cols` are all the columns of `aux_table` except `aux_key`."

    # Check no cols
    agg_joiner = AggJoiner(
        aux_table=main_table, operations="mode", key=["movieId", "userId"]
    )
    agg_joiner.fit(main_table)
    assert agg_joiner._cols == ["rating", "genre"]


def test_agg_joiner_correct_cols(df_module, main_table):
    "Check that expected `cols` parameters for the `AggJoiner` are working."

    # Check one col
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations="mean",
        key=["movieId", "userId"],
        cols=["rating"],
    )
    agg_joiner.fit(main_table)
    assert agg_joiner._cols == ["rating"]


def test_agg_joiner_wrong_cols(main_table):
    "Check that providing a column that's not in `aux_table` does not work."

    # Check missing agg or keys cols in tables
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations="mean",
        key="userId",
        cols="unknown_col",
    )
    match = r"(columns cannot be used because they do not exist)"
    with pytest.raises(ValueError, match=match):
        agg_joiner.fit(main_table)


def test_agg_joiner_input_multiple_tables(main_table):
    "Check that providing too many auxiliary tables in `AggJoiner` raises an error."

    # Check too many aux_table
    agg_joiner = AggJoiner(
        aux_table=[main_table, main_table],
        operations="mean",
        main_key="userId",
        aux_key=["userId", "userId"],
        cols=[["rating"], ["rating"]],
    )
    error_msg = (
        r"(?=.*must be a dataframe or the string 'X')"
        r"(?=.*use the MultiAggJoiner instead)"
    )
    with pytest.raises(ValueError, match=error_msg):
        agg_joiner.fit_transform(main_table)


def test_agg_joiner_correct_operations_input(main_table):
    "Check that expected `operations` parameters for the `AggJoiner` are working."

    # Check invariant operations input
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations=["min", "max", "mode"],
        key="userId",
        cols=["rating", "genre"],
    )
    agg_joiner.fit(main_table)
    assert agg_joiner._operations == ["min", "max", "mode"]


def test_agg_joiner_not_supported_operations(main_table):
    "Check that calling an unsupported operation raises an error."

    # Check not supported operations
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations=["nunique", "mode"],
        key="userId",
        cols=["rating", "genre"],
    )
    match = r"(?=.*`operations` options are)"
    with pytest.raises(ValueError, match=match):
        agg_joiner.fit(main_table)


def test_agg_joiner_wrong_string_placeholder(main_table):
    "Check that `aux_table='Y'` is not a valid string placeholder."

    agg_joiner = AggJoiner(
        aux_table="Y",
        operations="mean",
        key="userId",
        cols="genre",
    )
    with pytest.raises(ValueError, match=r"(?=.*dataframe)(?=.*'X')"):
        agg_joiner.fit(main_table)


def test_agg_joiner_get_feature_names_out(main_table):
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations="count",
        key="userId",
        cols="genre",
    )
    with pytest.raises(NotFittedError):
        agg_joiner.get_feature_names_out()

    agg_joiner.fit(main_table)
    assert agg_joiner.get_feature_names_out() == [
        "userId",
        "movieId",
        "rating",
        "genre",
        "genre_count",
    ]


def test_agg_joiner_not_fitted_dataframe(df_module, main_table):
    """
    Check that calling `transform` on a dataframe not containing the columns
    seen during `fit` raises an error.
    """

    not_main_table = df_module.make_dataframe(
        {"wrong": [1, 2, 3], "dataframe": [4, 5, 6]}
    )

    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations="mean",
        key="userId",
        cols="rating",
    )
    agg_joiner.fit(main_table)
    error_msg = r"Columns of dataframes passed to fit\(\) and transform\(\) differ"
    with pytest.raises(ValueError, match=error_msg):
        agg_joiner.transform(not_main_table)


def test_agg_joiner_duplicate_columns(main_table):
    joiner = AggJoiner(
        aux_table=main_table, operations="mean", key="userId", cols="rating"
    )
    X = sbd.with_columns(main_table, rating_mean=sbd.col(main_table, "rating"))
    out_1 = joiner.fit_transform(X)
    out_2 = joiner.transform(X)
    assert sbd.column_names(out_1) == sbd.column_names(out_2)


@pytest.fixture
def y_df(df_module):
    return df_module.make_dataframe({"rating": [4.1, 4.1, 4.1, 3.1, 2.1, 4.1]})


@pytest.fixture(params=["df", "named_column", "1d_array", "2d_array"])
def y_col_name(df_module, request):
    input_type = request.param
    y = df_module.make_dataframe({"rating": [4.1, 4.1, 4.1, 3.1, 2.1, 4.1]})
    if input_type == "df":
        return (y, "rating")
    if input_type == "named_column":
        return (sbd.col(y, "rating"), "rating")
    if input_type == "1d_array":
        return (np.array([4.1, 4.1, 4.1, 3.1, 2.1, 4.1]), "y_0")
    if input_type == "2d_array":
        return (
            np.asarray([sbd.to_numpy(c) for c in sbd.to_column_list(y)], dtype=float).T,
            "y_0",
        )


def test_agg_target_simple_fit_transform(df_module, main_table, y_col_name):
    y, col_name = y_col_name
    agg_target = AggTarget(main_key="userId", operations="mean", suffix="_user")
    main_transformed = agg_target.fit_transform(main_table, y)

    main_transformed_expected = df_module.make_dataframe(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.1, 4.1, 4.1, 3.1, 2.1, 4.1],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
            f"{col_name}_mean_user": [4.1, 4.1, 4.1, 3.1, 3.1, 3.1],
        }
    )

    if df_module.description == "pandas-nullable-dtypes":
        main_transformed = sbd.pandas_convert_dtypes(main_transformed)
    df_module.assert_frame_equal(main_transformed, main_transformed_expected)


def test_agg_target_unnamed_column():
    main_table = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "rating": [4.1, 4.1, 4.1, 3.1, 2.1, 4.1],
        }
    )
    y = pd.Series([4.1, 4.1, 4.1, 3.1, 2.1, 4.1], name=None)

    agg_target = AggTarget(main_key="userId", operations="mean", suffix="_user")
    main_transformed = agg_target.fit_transform(main_table, y)
    main_transformed_expected = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "rating": [4.1, 4.1, 4.1, 3.1, 2.1, 4.1],
            "y_0_mean_user": [4.1, 4.1, 4.1, 3.1, 3.1, 3.1],
        }
    )

    pandas.testing.assert_frame_equal(main_transformed, main_transformed_expected)


@pytest.fixture(params=["df", "array"])
def y_2_col_names(df_module, request):
    input_type = request.param
    y = df_module.make_dataframe(
        {
            "a": [10.1, 20.1, 30.1, 40.1, 50.1, 60.1],
            "b": [60.1, 50.1, 40.1, 30.1, 20.1, 10.1],
        }
    )
    if input_type == "df":
        return (y, "a", "b")
    if input_type == "array":
        return (
            np.asarray([sbd.to_numpy(c) for c in sbd.to_column_list(y)], dtype=float).T,
            "y_0",
            "y_1",
        )


def test_agg_target_multiple_columns(df_module, main_table, y_2_col_names):
    y, col_name_a, col_name_b = y_2_col_names
    agg_target = AggTarget(main_key="userId", operations="mean", suffix="_user")
    main_transformed = agg_target.fit_transform(main_table, y)

    main_transformed_expected = df_module.make_dataframe(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.1, 4.1, 4.1, 3.1, 2.1, 4.1],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
            f"{col_name_a}_mean_user": [20.1, 20.1, 20.1, 50.1, 50.1, 50.1],
            f"{col_name_b}_mean_user": [50.1, 50.1, 50.1, 20.1, 20.1, 20.1],
        }
    )
    if df_module.description == "pandas-nullable-dtypes":
        main_transformed = sbd.pandas_convert_dtypes(main_transformed)
    df_module.assert_frame_equal(main_transformed, main_transformed_expected)


def test_agg_target_multiple_main_key(df_module, main_table, y_df):
    agg_target = AggTarget(
        main_key=["userId", "genre"],
        operations="max",
    )
    main_transformed = agg_target.fit_transform(main_table, y_df)
    assert sbd.column_names(main_transformed) == [
        "userId",
        "movieId",
        "rating",
        "genre",
        "rating_max_target",
    ]
    if df_module.description == "pandas-nullable-dtypes":
        main_transformed = sbd.pandas_convert_dtypes(main_transformed)
    y_expected = df_module.make_column(
        "rating_max_target", [4.1, 4.1, 4.1, 4.1, 2.1, 4.1]
    )
    df_module.assert_column_equal(
        sbd.col(main_transformed, "rating_max_target"), y_expected
    )


def test_agg_target_wrong_target_size(main_table, y_df):
    agg_target = AggTarget(
        main_key="userId",
        operations="count",
        suffix="_user",
    )

    match = r"(?=.*length)(?=.*match)"
    with pytest.raises(ValueError, match=match):
        agg_target.fit(main_table, sbd.col(y_df, "rating")[:2])


def test_agg_target_wrong_operations(main_table, y_df):
    "Check that a useful error is raised when `operations` is not supported"

    # Test operations is string not in list
    agg_target = AggTarget(
        main_key="userId",
        operations="wrong_op",
    )
    with pytest.raises(
        ValueError,
        match=r"(`operations` options are)",
    ):
        agg_target.fit(main_table, y_df)

    # Test operations is None
    agg_target = AggTarget(
        main_key="userId",
        operations=None,
    )
    with pytest.raises(
        ValueError,
        match=r"(`operations` must be a string or an iterable of strings)",
    ):
        agg_target.fit(main_table, y_df)

    # Test operations is int
    agg_target = AggTarget(
        main_key="userId",
        operations=2,
    )
    with pytest.raises(
        ValueError,
        match=r"(`operations` must be a string or an iterable of strings)",
    ):
        agg_target.fit(main_table, y_df)


def test_agg_target_too_many_suffixes(main_table, y_df):
    "Check that providing more than one `suffix` for the `AggTarget` raises an error."

    # Check inconsistent number of suffixes
    agg_target = AggTarget(
        main_key="userId",
        operations="mean",
        suffix=["_user", "_movie", "_tag"],
    )
    with pytest.raises(ValueError, match=r"(?='suffix' must be a string.*)"):
        agg_target.fit(main_table, y_df)


def test_agg_target_get_feature_names_out(main_table, y_df):
    agg_target = AggTarget(
        main_key="userId",
        operations="count",
    )
    with pytest.raises(NotFittedError):
        agg_target.get_feature_names_out()

    agg_target.fit(main_table, y_df)
    assert agg_target.get_feature_names_out() == [
        "userId",
        "movieId",
        "rating",
        "genre",
        "rating_count_target",
    ]


def test_agg_target_non_dataframe_X(y_df):
    agg_target = AggTarget(main_key="userId", operations="mean")
    with pytest.raises(
        TypeError, match=r"Only pandas and polars DataFrames are supported"
    ):
        agg_target.fit("should_be_a_dataframe", y_df)


def test_agg_target_non_dataframe_y(main_table):
    agg_target = AggTarget(main_key="userId", operations="mean")
    with pytest.raises(
        TypeError, match=r"`y` must be a dataframe, a series or a numpy array"
    ):
        agg_target.fit(main_table, "should_be_array_like")


def test_agg_target_duplicate_columns(main_table, y_df):
    agg_target = AggTarget(main_key="userId", operations="mean")
    X = sbd.with_columns(main_table, rating_mean_target=sbd.col(main_table, "rating"))
    out_1 = agg_target.fit_transform(X, y_df)
    out_2 = agg_target.transform(X)
    assert sbd.column_names(out_1) == sbd.column_names(out_2)


def test_error_perform_groupby():
    # Make codecov happy
    with pytest.raises(TypeError, match="Expecting a Pandas or Polars DataFrame"):
        perform_groupby(np.array([1]), key=None, cols_to_agg=None, operations=None)
