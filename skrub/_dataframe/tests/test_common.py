from skrub._dataframe import _common as ns


def test_column_names(empty_df):
    data = {"a": [0], "b c": [0]}
    df = ns.dataframe_from_dict(empty_df, data)
    assert ns.column_names(df) == ["a", "b c"]
