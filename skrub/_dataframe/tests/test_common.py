from skrub._dataframe import skrubns, std, stdns


def test_std(px):
    df = px.DataFrame({"A": [1, 2]})
    assert hasattr(std(df), "dataframe")
    assert hasattr(stdns(df), "dataframe_from_columns")
    ns = skrubns(df)
    s = ns.make_series([1, 2], name="A")
    assert hasattr(std(s), "column")
    assert hasattr(stdns(s), "dataframe_from_columns")
