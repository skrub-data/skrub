import inspect
import pickle
import types

import numpy as np
import pandas as pd
import pytest

from skrub import _dataframe as sbd
from skrub import selectors as s
from skrub.selectors._base import _select_col_names


def test_repr():
    """
    >>> from skrub import selectors as s
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
    >>> s.has_nulls()
    has_nulls(0.0)
    >>> s.has_dtype('x', 'y')
    has_dtype('x', 'y')
    """


def test_glob(df_module):
    df = df_module.example_dataframe
    assert s.glob("*").expand(df) == s.all().expand(df)
    assert s.glob("xxx").expand(df) == []
    assert (s.glob("[Ii]nt-*") | s.glob("?loat-col")).expand(df) == [
        "int-col",
        "int-not-null-col",
        "float-col",
    ]


def test_regex(df_module):
    df = df_module.example_dataframe
    assert (s.regex("int-.*") | s.regex("float-") | s.regex("date-$")).expand(df) == [
        "int-col",
        "int-not-null-col",
        "float-col",
    ]


def test_dtype_selectors(df_module):
    df = df_module.example_dataframe
    cat_col = sbd.rename(sbd.to_categorical(sbd.col(df, "str-col")), "cat-col")
    df = sbd.make_dataframe_like(df, sbd.to_column_list(df) + [cat_col])
    assert s.numeric().expand(df) == ["int-col", "int-not-null-col", "float-col"]
    assert (s.numeric() | s.boolean()).expand(df) == [
        "int-col",
        "int-not-null-col",
        "float-col",
        "bool-col",
        "bool-not-null-col",
    ]
    if df_module.description == "pandas-numpy-dtypes":
        int_cols = ["int-not-null-col"]
        float_cols = ["int-col", "float-col"]
    else:
        int_cols = ["int-col", "int-not-null-col"]
        float_cols = ["float-col"]
    assert s.integer().expand(df) == int_cols
    assert s.float().expand(df) == float_cols
    assert s.string().expand(df) == ["str-col"]
    assert s.categorical().expand(df) == ["cat-col"]
    if df_module.name == "polars":
        assert s.any_date().expand(df) == ["datetime-col", "date-col"]
    else:
        # pandas doesn't have a 'date' dtype, only datetime
        assert df_module.name == "pandas"
        assert s.any_date().expand(df) == ["datetime-col"]


def test_has_dtype(df_module):
    df = df_module.make_dataframe(
        {
            "int-col": [1, 2],
            "float-col": [1.5, 2.5],
            "list-col": [[1, 2], [3]],
        }
    )
    int_dtype = sbd.dtype(sbd.col(df, "int-col"))
    float_dtype = sbd.dtype(sbd.col(df, "float-col"))
    list_dtype = sbd.dtype(sbd.col(df, "list-col"))

    assert s.has_dtype(int_dtype).expand(df) == ["int-col"]
    assert s.has_dtype(float_dtype).expand(df) == ["float-col"]
    assert s.has_dtype(list_dtype).expand(df) == ["list-col"]
    assert s.has_dtype(int_dtype, list_dtype).expand(df) == ["int-col", "list-col"]
    assert s.has_dtype("definitely-not-a-dtype").expand(df) == []


def test_dtype_pandas_object():
    # Testing for behavior with object and string columns
    df = pd.DataFrame({"string-object": ["foo", "bar"], "object-object": ["baz", 42]})

    assert s.string().expand(df) == ["string-object"]


def test_cardinality_below(df_module, monkeypatch):
    df = df_module.example_dataframe
    assert s.cardinality_below(3).expand(df) == ["bool-col", "bool-not-null-col"]
    assert s.cardinality_below(4).expand(df) == (
        s.all() - "date-col" - "int-not-null-col"
    ).expand(df)
    assert s.cardinality_below(5).expand(df) == s.all().expand(df)

    def bad_n_unique(c):
        raise ValueError()

    monkeypatch.setattr(sbd, "n_unique", bad_n_unique)
    assert s.cardinality_below(5).expand(df) == []


def test_has_nulls(df_module):
    df = df_module.make_dataframe(dict(a=[0, 1, 2], b=[0, None, 2], c=["a", "b", None]))
    assert s.has_nulls().expand(df) == ["b", "c"]


def test_has_nulls_proportion(df_module):
    df = df_module.make_dataframe(
        dict(a=[0, 1, 2, None], b=[0, None, 2, None], c=["a", None, None, None])
    )
    assert s.has_nulls(proportion=0).expand(df) == ["a", "b", "c"]
    assert s.has_nulls(proportion=0.20).expand(df) == ["a", "b", "c"]
    assert s.has_nulls(proportion=0.45).expand(df) == ["b", "c"]
    assert s.has_nulls(proportion=0.70).expand(df) == ["c"]
    assert s.has_nulls(proportion=1.0).expand(df) == []


def test_has_nulls_proportion_wrong(df_module):
    df = df_module.make_dataframe(
        dict(a=[0, 1, 2, None], b=[0, None, 2, None], c=["a", None, None, None])
    )
    with pytest.raises(ValueError, match="should be a number in the range"):
        s.has_nulls(proportion=None).expand(df)

    with pytest.raises(ValueError, match="should be a number in the range"):
        s.has_nulls(proportion="0.0").expand(df)


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
        s.filter(_filt_col, "int-"),
        s.filter_names(_filt_col_name, "int-"),
        s.glob("int-*"),
        s.regex("^int-.*$"),
        s.cardinality_below(4),
        s.has_dtype(sbd.dtype(sbd.col(df, "float-col"))),
        s.cols("int-col", "float-col"),
    ]:
        unpickled = pickle.loads(pickle.dumps(selector))
        assert unpickled.expand(df) == selector.expand(df)
    # unless of course the ones that have been initialized with an un-picklable
    # attribute
    with pytest.raises(Exception):
        pickle.dumps(s.filter(lambda _: False))


def test_error_select_col_names():
    with pytest.raises(TypeError, match="Expecting a Pandas or Polars DataFrame"):
        _select_col_names(np.array([1]), col_names=None)
