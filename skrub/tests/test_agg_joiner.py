import re

import pandas as pd
import pytest

from skrub import _dataframe as sbd
from skrub._agg_joiner import AggJoiner, AggTarget, aggregate


# TODO: rename tests to correspond to AggJoiner / AggTarget
# TODO: parametrize the fixture with df_module
# TODO: check empty col
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


def test_aggregate_single_operation(df_module, main_table):
    main_table = df_module.DataFrame(main_table)
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
    expected = df_module.make_dataframe({"userId": [1, 2], "rating_mean": [4.0, 3.0]})
    df_module.assert_frame_equal(
        sbd.pandas_convert_dtypes(aggregated), sbd.pandas_convert_dtypes(expected)
    )

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
    df_module.assert_frame_equal(
        sbd.pandas_convert_dtypes(aggregated), sbd.pandas_convert_dtypes(expected)
    )


def test_aggregate_multiple_operations(df_module, main_table):
    main_table = df_module.DataFrame(main_table)
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
        {"userId": [1, 2], "rating_mean": [4.0, 3.0], "rating_sum": [12.0, 9.0]}
    )
    df_module.assert_frame_equal(
        sbd.pandas_convert_dtypes(aggregated), sbd.pandas_convert_dtypes(expected)
    )

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
        {"userId": [1, 2], "rating_sum": [12.0, 9.0], "rating_mean": [4.0, 3.0]}
    )
    df_module.assert_frame_equal(
        sbd.pandas_convert_dtypes(aggregated), sbd.pandas_convert_dtypes(expected)
    )


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
def test_simple_fit_transform(df_module, main_table, use_X_placeholder):
    "Check the general behaviour of the `AggJoiner`."
    X = df_module.DataFrame(main_table)
    aux = X if not use_X_placeholder else "X"

    agg_joiner_user = AggJoiner(
        aux_table=aux,
        operations="mode",
        aux_key="userId",
        main_key="userId",
        cols="genre",
        suffix="_user",
    )

    main_user = agg_joiner_user.fit_transform(X)

    expected_user = df_module.make_dataframe(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
            "genre_mode_user": ["drama", "drama", "drama", "sf", "sf", "sf"],
        }
    )
    df_module.assert_frame_equal(
        sbd.pandas_convert_dtypes(main_user), sbd.pandas_convert_dtypes(expected_user)
    )

    agg_joiner_movie = AggJoiner(
        aux_table=aux,
        operations="mean",
        aux_key="movieId",
        main_key="movieId",
        cols="rating",
        suffix="_movie",
    )

    main_movie = agg_joiner_movie.fit_transform(X)

    expected_movie = df_module.make_dataframe(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
            "rating_mean_movie": [4.0, 4.0, 3.0, 3.0, 3.0, 4.0],
        }
    )

    df_module.assert_frame_equal(
        sbd.pandas_convert_dtypes(main_movie), sbd.pandas_convert_dtypes(expected_movie)
    )


def test_wrong_operations(df_module, main_table):
    "Check that a useful error is raised when `operations` is not supported"
    main_table = df_module.DataFrame(main_table)

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
        match=r"(`operations` must be string or an iterable of strings)",
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
        match=r"(`operations` must be string or an iterable of strings)",
    ):
        agg_joiner.fit(main_table)


def test_correct_keys(df_module, main_table):
    "Check that expected `key` parameters for the `AggJoiner` are working."
    main_table = df_module.DataFrame(main_table)

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


def test_wrong_keys(df_module, main_table):
    "Check that wrong `key` parameters for the `AggJoiner` raise an error."
    main_table = df_module.DataFrame(main_table)

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


def test_default_suffix(df_module, main_table):
    "Check that the default `suffix` of `AggJoiner` is ''."
    main_table = df_module.DataFrame(main_table)

    # Check no suffix
    agg_joiner = AggJoiner(
        aux_table=main_table, operations="mode", key="userId", cols=["rating", "genre"]
    )
    agg_joiner.fit(main_table)
    assert agg_joiner.suffix == ""


def test_too_many_suffixes(df_module, main_table):
    "Check that providing more than one `suffix` for the `AggJoiner` raises an error."
    main_table = df_module.DataFrame(main_table)

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


def test_duplicate_col_name_after_suffix(df_module, main_table):
    "Check that ``__skrub_<random string>__`` is added for duplicate column names."
    main_table = df_module.DataFrame(main_table)

    main_table = sbd.with_columns(
        main_table, **{"rating_mean": [4.0, 4.0, 4.0, 3.0, 3.0, 3.0]}
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


def test_default_cols(df_module, main_table):
    "Check that by default, `cols` are all the columns of `aux_table` except `aux_key`."
    main_table = df_module.DataFrame(main_table)

    # Check no cols
    agg_joiner = AggJoiner(
        aux_table=main_table, operations="mode", key=["movieId", "userId"]
    )
    agg_joiner.fit(main_table)
    agg_joiner._cols == ["rating", "genre"]


def test_correct_cols(df_module, main_table):
    "Check that expected `cols` parameters for the `AggJoiner` are working."
    main_table = df_module.DataFrame(main_table)

    # Check one col
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations="mean",
        key=["movieId", "userId"],
        cols=["rating"],
    )
    agg_joiner.fit(main_table)
    agg_joiner._cols == ["rating"]


def test_wrong_cols(df_module, main_table):
    "Check that providing a column that's not in `aux_table` does not work."
    main_table = df_module.DataFrame(main_table)

    # Check missing agg or keys cols in tables
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations="mean",
        key="userId",
        cols="unknown_col",
    )
    match = r"(The following columns are requested for selection)"
    with pytest.raises(ValueError, match=match):
        agg_joiner.fit(main_table)


def test_input_multiple_tables(df_module, main_table):
    "Check that providing too many auxiliary tables in `AggJoiner` raises an error."
    main_table = df_module.DataFrame(main_table)

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


def test_correct_operations_input(df_module, main_table):
    "Check that expected `operations` parameters for the `AggJoiner` are working."
    main_table = df_module.DataFrame(main_table)

    # Check invariant operations input
    agg_joiner = AggJoiner(
        aux_table=main_table,
        operations=["min", "max", "mode"],
        key="userId",
        cols=["rating", "genre"],
    )
    agg_joiner.fit(main_table)
    assert agg_joiner._operations == ["min", "max", "mode"]


def test_not_supported_operations(df_module, main_table):
    "Check that calling an unsupported operation raises an error."
    main_table = df_module.DataFrame(main_table)

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


def test_wrong_string_placeholder(df_module, main_table):
    "Check that `aux_table='Y'` is not a valid string placeholder."
    main_table = df_module.DataFrame(main_table)

    agg_joiner = AggJoiner(
        aux_table="Y",
        operations="mean",
        key="userId",
        cols="genre",
    )
    with pytest.raises(ValueError, match=r"(?=.*dataframe)(?=.*'X')"):
        agg_joiner.fit(main_table)


def test_not_fitted_dataframe(df_module, main_table):
    """
    Check that calling `transform` on a dataframe not containing the columns
    seen during `fit` raises an error.
    """
    main_table = df_module.DataFrame(main_table)
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


y = pd.DataFrame(dict(rating=[4.0, 4.0, 4.0, 3.0, 2.0, 4.0]))


@pytest.mark.parametrize(
    "y, col_name",
    [
        (y, "rating"),
        (y["rating"], "rating"),
        (y.values, "y_0"),
        (y.values.tolist(), "y_0"),
        (y["rating"].rename(None), "y_0"),
    ],
)
def test_agg_target(main_table, y, col_name):
    agg_target = AggTarget(main_key="userId", suffix="_user", operations="mean")
    main_transformed = agg_target.fit_transform(main_table, y)

    main_transformed_expected = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
            f"{col_name}_mean_user": [4.0, 4.0, 4.0, 3.0, 3.0, 3.0],
        }
    )
    pd.testing.assert_frame_equal(main_transformed, main_transformed_expected)


def test_agg_target_missing_operations(main_table):
    agg_target = AggTarget(
        main_key="userId",
        suffix="_user",
    )

    # y is continuous
    y = pd.DataFrame(dict(rating=[4.0, 4.1, 4.2, 4.3, 4.4, 4.5]))
    agg_target.fit(main_table, y)
    assert agg_target.operations_ == ["mean"]

    # y is categorical
    y = pd.DataFrame(dict(rating=["1", "2", "3", "1", "2", "3"]))
    agg_target.fit(main_table, y)
    assert agg_target.operations_ == ["mode"]


def test_agg_target_check_input(main_table):
    agg_target = AggTarget(
        main_key="userId",
        suffix="_user",
    )
    match = r"(?=.*X must be a dataframe)"
    with pytest.raises(TypeError, match=match):
        agg_target.fit(main_table.values, y)

    match = r"(?=.*length)(?=.*match)"
    with pytest.raises(ValueError, match=match):
        agg_target.fit(main_table, y["rating"][:2])


def test_duplicate_columns(df_module, main_table):
    main_table = df_module.DataFrame(main_table)
    joiner = AggJoiner(
        aux_table=main_table, operations="mean", key="userId", cols="rating"
    )
    X = sbd.with_columns(main_table, rating_mean=sbd.col(main_table, "rating"))
    out_1 = joiner.fit_transform(X)
    out_2 = joiner.transform(X)
    assert sbd.column_names(out_1) == sbd.column_names(out_2)


def test_duplicate_columns_target(main_table):
    joiner = AggTarget(main_key="userId", operations="mean")
    y = sbd.col(main_table, "rating")
    X = sbd.with_columns(main_table, rating_mean_target=sbd.col(main_table, "rating"))
    out_1 = joiner.fit_transform(X, y)
    out_2 = joiner.transform(X)
    assert sbd.column_names(out_1) == sbd.column_names(out_2)
