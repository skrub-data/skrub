import pytest

from skrub import _dataframe as sbd
from skrub import _selectors as s


def test_repr():
    """
    >>> from skrub import _selectors as s
    >>> s.all()
    all()
    >>> s.all() - ["ID", "Name"]
    (all() - cols('ID', 'Name'))
    >>> s.cols("ID", "Name") & "ID"
    (cols('ID', 'Name') & cols('ID'))
    >>> s.filter_names(lambda n: 'a' in n) ^ s.filter(lambda c: c[2] == 3)
    (filter_names(<lambda>) ^ filter(<lambda>))
    >>> ~s.all()
    (~all())
    >>> s.Filter(lambda c, x: c[2] == x, args=(3,), name='my_filter')
    my_filter(3)
    >>> s.Filter(lambda c, x: c[2] == x, args=(3,), selector_repr='my_filter()')
    my_filter()
    >>> s.NameFilter(lambda c_n, n: c_n.lower() == n,
    ...              args=('col',),
    ...              selector_repr='lower_check()')
    lower_check()
    """


def test_make_selector():
    assert s.make_selector(sel := s.all()) is sel
    assert s.make_selector(cols := ["a", "one two"]).columns == cols
    assert s.make_selector(col := "one two").columns == [col]
    with pytest.raises(ValueError, match="Selector not understood"):
        s.make_selector(0)


def test_select(df_module):
    df = df_module.example_dataframe
    df_module.assert_frame_equal(
        s.select(df, s.cols("float-col", "str-col")), df[["float-col", "str-col"]]
    )


def test_all(df_module):
    assert s.all().expand(df_module.example_dataframe) == [
        "int-col",
        "int-not-null-col",
        "float-col",
        "str-col",
        "bool-col",
        "bool-not-null-col",
        "datetime-col",
        "date-col",
    ]
    assert s.all().expand(df_module.empty_dataframe) == []


def test_cols(df_module):
    expanded = s.cols("bool-col", "int-col").expand(df_module.example_dataframe)
    assert expanded == ["bool-col", "int-col"]
    expanded = s.cols().expand(df_module.example_dataframe)
    assert expanded == []
    expanded = s.cols().expand(df_module.empty_dataframe)
    assert expanded == []
    with pytest.raises(ValueError, match=".*Found non-string.*"):
        s.cols(2, "2")


def test_missing_cols_error(df_module):
    with pytest.raises(ValueError, match="The following columns .*missing.*"):
        s.cols("a", "b").expand(df_module.example_dataframe)


def test_missing_cols_in_expr_no_error(df_module):
    assert (s.cols("a", "b") & s.all()).expand(df_module.example_dataframe) == []


def test_cols_from_list(df_module):
    df = df_module.example_dataframe
    assert (["bool-col", "int-col"] & s.all()).expand(df) == ["int-col", "bool-col"]
    assert (s.all() & ["bool-col", "int-col"]).expand(df) == ["int-col", "bool-col"]
    assert (s.cols("date-col", "bool-col") & ["bool-col", "int-col"]).expand(df) == [
        "bool-col"
    ]


def test_cols_from_name(df_module):
    df = df_module.example_dataframe
    assert ("bool-col" & s.all()).expand(df) == ["bool-col"]
    assert (s.cols("date-col") | "bool-col").expand(df) == ["bool-col", "date-col"]


def test_filter(df_module):
    df = df_module.example_dataframe
    expanded = s.filter(lambda c: 4.5 in sbd.to_list(c)).expand(df)
    assert expanded == ["float-col"]
    expanded = s.filter(lambda c, v: v in sbd.to_list(c), 4.5).expand(df)
    assert expanded == ["float-col"]


def test_filter_names(df_module):
    df = df_module.example_dataframe
    expanded = s.filter_names(str.endswith, "t-col").expand(df)
    assert expanded == ["int-col", "float-col"]


def test_inv(df_module):
    df = df_module.example_dataframe
    assert s.inv(["int-col", "float-col", "str-col"]).expand(df) == [
        "int-not-null-col",
        "bool-col",
        "bool-not-null-col",
        "datetime-col",
        "date-col",
    ]
    assert (~s.all()).expand(df) == []
    assert (~s.cols()).expand(df) == s.all().expand(df)


def test_or(df_module):
    df = df_module.example_dataframe
    assert (~s.all() | "int-col" | "float-col").expand(df) == ["int-col", "float-col"]
    assert (("abc" | s.all()) & "int-col").expand(df) == ["int-col"]


def test_and(df_module):
    df = df_module.example_dataframe
    assert ("int-col" & s.cols("float-col") | "float-col").expand(df) == ["float-col"]


def test_sub(df_module):
    df = df_module.example_dataframe
    assert (s.all() - "int-col").expand(df) == (s.inv("int-col")).expand(df)
    assert (["int-col", "float-col"] - s.all()).expand(df) == []


def test_xor(df_module):
    df = df_module.example_dataframe
    assert (("int-col", "float-col", "xx") ^ s.cols("int-col", "date-col")).expand(
        df
    ) == ["float-col", "date-col"]
    assert (s.cols("int-col", "float-col", "xx") ^ ("int-col", "date-col")).expand(
        df
    ) == ["float-col", "date-col"]


def test_short_circuit(df_module):
    df = df_module.example_dataframe

    def sel(col):
        raise ValueError("problem")

    assert (s.all() | s.filter(sel)).expand(df) == s.all().expand(df)
    assert (~s.all() & s.filter(sel)).expand(df) == []
    with pytest.raises(ValueError, match="problem"):
        (s.filter(sel) | s.all()).expand(df)
