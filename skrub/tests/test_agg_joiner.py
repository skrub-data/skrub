import pandas as pd
import pytest

from skrub import _dataframe as sbd
from skrub._agg_joiner import AggJoiner, AggTarget, split_num_categ_operations
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


def test_split_num_categ_operations():
    "Check that the operations are correctly separated."

    # Check split ops
    num_ops, categ_ops = split_num_categ_operations(
        ["mean", "std", "min", "mode", "max", "sum"]
    )
    assert num_ops == ["mean", "std", "min", "max", "sum"]
    assert categ_ops == ["mode"]


@pytest.mark.parametrize("use_X_placeholder", [False, True])
def test_simple_fit_transform(df_module, main_table, use_X_placeholder):
    "Check the general behaviour of the `AggJoiner`."
    X = df_module.DataFrame(main_table)
    aux = X if not use_X_placeholder else "X"

    agg_joiner_user = AggJoiner(
        aux_table=aux,
        aux_key="userId",
        main_key="userId",
        cols=["rating", "genre"],
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
            "rating_mean_user": [4.0, 4.0, 4.0, 3.0, 3.0, 3.0],
        }
    )
    df_module.assert_frame_equal(
        sbd.pandas_convert_dtypes(main_user), sbd.pandas_convert_dtypes(expected_user)
    )

    agg_joiner_movie = AggJoiner(
        aux_table=aux,
        aux_key="movieId",
        main_key="movieId",
        cols=["rating"],
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


def test_correct_keys(df_module, main_table):
    "Check that expected `key` parameters for the `AggJoiner` are working."
    main_table = df_module.DataFrame(main_table)

    # Check only key
    agg_joiner = AggJoiner(
        aux_table=main_table,
        key="userId",
        cols=["rating", "genre"],
    )
    agg_joiner.fit(main_table)
    assert agg_joiner._main_key == ["userId"]
    assert agg_joiner._aux_key == ["userId"]

    # Check multiple key
    agg_joiner = AggJoiner(
        aux_table=main_table,
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
        key="userId",
        main_key="userId",
        cols=["rating", "genre"],
    )
    with pytest.raises(ValueError, match=r"(?=.*not a combination of both.)"):
        agg_joiner.fit(main_table)

    # Check providing key and extra aux_key
    agg_joiner = AggJoiner(
        aux_table=main_table,
        key="userId",
        aux_key="userId",
        cols=["rating", "genre"],
    )
    with pytest.raises(ValueError, match=r"(?=.*not a combination of both.)"):
        agg_joiner.fit(main_table)

    # Check main_key doesn't exist in table
    agg_joiner = AggJoiner(
        aux_table=main_table,
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
        aux_table=main_table,
        key="userId",
        cols=["rating", "genre"],
    )
    agg_joiner.fit(main_table)
    assert agg_joiner.suffix == ""


def test_too_many_suffixes(df_module, main_table):
    "Check that providing more than one `suffix` for the `AggJoiner` raises an error."
    main_table = df_module.DataFrame(main_table)

    # Check inconsistent number of suffixes
    agg_joiner = AggJoiner(
        aux_table=main_table,
        key="userId",
        cols="rating",
        suffix=["_user", "_movie", "_tag"],
    )
    with pytest.raises(ValueError, match=r"(?='suffix' must be a string.*)"):
        agg_joiner.fit(main_table)


def test_default_cols(df_module, main_table):
    "Check that by default, `cols` are all the columns of `aux_table` except `aux_key`."
    main_table = df_module.DataFrame(main_table)

    # Check no cols
    agg_joiner = AggJoiner(
        aux_table=main_table,
        key=["movieId", "userId"],
    )
    agg_joiner.fit(main_table)
    agg_joiner._cols == ["rating", "genre"]


def test_correct_cols(df_module, main_table):
    "Check that expected `cols` parameters for the `AggJoiner` are working."
    main_table = df_module.DataFrame(main_table)

    # Check one col
    agg_joiner = AggJoiner(
        aux_table=main_table,
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
        key="userId",
        cols="unknown_col",
    )
    match = r"(?=.*columns cannot be used because they do not exist)"
    with pytest.raises(ValueError, match=match):
        agg_joiner.fit(main_table)


def test_input_multiple_tables(df_module, main_table):
    "Check that providing too many auxiliary tables in `AggJoiner` raises an error."
    main_table = df_module.DataFrame(main_table)

    # Check too many aux_table
    agg_joiner = AggJoiner(
        aux_table=[main_table, main_table],
        main_key="userId",
        aux_key=["userId", "userId"],
        cols=[["rating"], ["rating"]],
    )
    error_msg = (
        r"(?=.*must be a dataframe or the string 'X')"
        r"(?=.*use the MultiAggJoiner instead)"
    )
    with pytest.raises(TypeError, match=error_msg):
        agg_joiner.fit_transform(main_table)


def test_default_operations(df_module, main_table):
    "Check that the default `operations` of `AggJoiner` are ['mean', 'mode']."
    main_table = df_module.DataFrame(main_table)

    # Check default operations
    agg_joiner = AggJoiner(
        aux_table=main_table,
        key="userId",
        cols=["rating", "genre"],
    )
    agg_joiner.fit(main_table)
    assert agg_joiner._operations == ["mean", "mode"]


def test_correct_operations_input(df_module, main_table):
    "Check that expected `operations` parameters for the `AggJoiner` are working."
    main_table = df_module.DataFrame(main_table)

    # Check invariant operations input
    agg_joiner = AggJoiner(
        aux_table=main_table,
        key="userId",
        cols=["rating", "genre"],
        operations=["min", "max", "mode"],
    )
    agg_joiner.fit(main_table)
    assert agg_joiner._operations == ["min", "max", "mode"]


@pytest.mark.skipif(not _POLARS_INSTALLED, reason="Polars is not available")
def test_polars_unavailable_operation(main_table):
    "Check that the 'value_counts' operation is not supported for Polars."
    import polars as pl

    agg_joiner = AggJoiner(
        aux_table="X",
        main_key="userId",
        aux_key="movieId",
        cols="rating",
        operations=["value_counts"],
    )
    with pytest.raises(ValueError, match=r"(?=.*value_counts)(?=.*supported)"):
        agg_joiner.fit(pl.DataFrame(main_table))


def test_not_supported_operations(df_module, main_table):
    "Check that calling an unsupported operation raises an error."
    main_table = df_module.DataFrame(main_table)

    # Check not supported operations
    agg_joiner = AggJoiner(
        aux_table=main_table,
        key="userId",
        cols=["rating", "genre"],
        operations=["nunique", "mode"],
    )
    match = r"(?=.*operations options are)"
    with pytest.raises(ValueError, match=match):
        agg_joiner.fit(main_table)


def test_wrong_string_placeholder(df_module, main_table):
    "Check that `aux_table='Y'` is not a valid string placeholder."
    main_table = df_module.DataFrame(main_table)

    agg_joiner = AggJoiner(
        aux_table="Y",
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
        key="userId",
    )
    agg_joiner.fit(main_table)
    error_msg = r"(?=.*columns cannot be used because they do not exist)"
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
    ],
)
def test_agg_target(main_table, y, col_name):
    agg_target = AggTarget(
        main_key="userId",
        suffix="_user",
        operation=["hist(2)", "value_counts"],
    )
    main_transformed = agg_target.fit_transform(main_table, y)

    main_transformed_expected = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 2],
            "movieId": [1, 3, 6, 318, 6, 1704],
            "rating": [4.0, 4.0, 4.0, 3.0, 2.0, 4.0],
            "genre": ["drama", "drama", "comedy", "sf", "comedy", "sf"],
            "index": [0, 0, 0, 1, 1, 1],
            f"{col_name}_(1.999, 3.0]_user": [0, 0, 0, 2, 2, 2],
            f"{col_name}_(3.0, 4.0]_user": [3, 3, 3, 1, 1, 1],
            f"{col_name}_2.0_user": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            f"{col_name}_3.0_user": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            f"{col_name}_4.0_user": [3.0, 3.0, 3.0, 1.0, 1.0, 1.0],
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
    assert agg_target.operation_ == ["mean"]

    # y is categorical
    y = pd.DataFrame(dict(rating=["1", "2", "3", "1", "2", "3"]))
    agg_target.fit(main_table, y)
    assert agg_target.operation_ == ["mode"]


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


def test_no_aggregation_exception(main_table):
    agg_target = AggTarget(
        main_key="userId",
        operation=[],
    )
    with pytest.raises(ValueError, match=r"(?=.*No aggregation)"):
        agg_target.fit(main_table, y)


def test_wrong_args_ops(main_table):
    agg_target = AggTarget(
        main_key="userId",
        operation="mean(2)",
    )
    with pytest.raises(ValueError, match=r"(?=.*'mean')(?=.*argument)"):
        agg_target.fit(main_table, y)
