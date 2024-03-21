import inspect
import pickle
import types

import pytest

from skrub import _dataframe as sbd
from skrub import _selectors as s


def test_repr():
    """
    >>> from skrub import _selectors as s
    >>> s.numeric() - s.boolean()
    (numeric() - boolean())
    >>> s.numeric() | s.glob("*_mm") - s.regex(r"^[ 0-9]+_mm$")
    (numeric() | (glob('*_mm') - regex('^[ 0-9]+_mm$')))
    >>> s.cardinality_below(30)
    cardinality_below(30)
    >>> s.string() | s.any_date() | s.categorical()
    ((string() | any_date()) | categorical())
    >>> s.float() & s.integer()
    (float() & integer())

    """


def test_glob(df_module):
    df = df_module.example_dataframe
    assert s.glob("*").expand(df) == s.all().expand(df)
    assert s.glob("xxx").expand(df) == []
    assert (s.glob("[Ii]nt-*") | s.glob("?loat-col")).expand(df) == [
        "int-col",
        "float-col",
    ]


def test_regex(df_module):
    df = df_module.example_dataframe
    assert (s.regex("int-.*") | s.regex("float-") | s.regex("date-$")).expand(df) == [
        "int-col",
        "float-col",
    ]


def test_dtype_selectors(df_module):
    df = df_module.example_dataframe
    cat_col = sbd.rename(sbd.to_categorical(sbd.col(df, "str-col")), "cat-col")
    df = sbd.make_dataframe_like(df, sbd.to_column_list(df) + [cat_col])
    assert s.numeric().expand(df) == ["int-col", "float-col"]
    assert (s.numeric() | s.boolean()).expand(df) == [
        "int-col",
        "float-col",
        "bool-col",
    ]
    assert s.integer().expand(df) == ["int-col"]
    assert s.float().expand(df) == ["float-col"]
    assert s.string().expand(df) == ["str-col"]
    assert s.categorical().expand(df) == ["cat-col"]


def test_cardinality_below(df_module, monkeypatch):
    df = df_module.example_dataframe
    assert s.cardinality_below(3).expand(df) == ["bool-col"]
    assert s.cardinality_below(4).expand(df) == (s.all() - "date-col").expand(df)
    assert s.cardinality_below(5).expand(df) == s.all().expand(df)

    def bad_n_unique(c):
        raise ValueError()

    monkeypatch.setattr(sbd, "n_unique", bad_n_unique)
    assert s.cardinality_below(5).expand(df) == []


@pytest.mark.parametrize("name", s.__all__)
def test_pickling_selectors_without_args(name, df_module):
    df = df_module.example_dataframe
    selector_func = getattr(s, name)
    if not isinstance(selector_func, types.FunctionType):
        return
    if len(inspect.signature(selector_func).parameters):
        return
    unpickled = pickle.loads(pickle.dumps(selector_func))
    assert unpickled().expand(df) == selector_func().expand(df)
    unpickled = pickle.loads(pickle.dumps(selector_func()))
    assert unpickled.expand(df) == selector_func().expand(df)


def _filt_col(col, pat):
    return pat in sbd.name(col)


def _filt_col_name(col_name, pat):
    return pat in col_name


def test_pickling_selectors_with_args(df_module):
    # pickling selectors should be fine ...
    df = df_module.example_dataframe
    for selector in [
        s.filter(_filt_col, args=("int-",)),
        s.filter_names(_filt_col_name, args=("int-",)),
        s.glob("int-*"),
        s.regex("^int-.*$"),
        s.cardinality_below(4),
        s.cols("int-col", "float-col"),
    ]:
        unpickled = pickle.loads(pickle.dumps(selector))
        assert unpickled.expand(df) == selector.expand(df)
    # unless of course the ones that have been initialized with an un-picklable
    # attribute
    with pytest.raises(Exception, match="Can't pickle"):
        pickle.dumps(s.filter(lambda _: False))
