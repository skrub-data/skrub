from inspect import ismodule

import numpy as np
import pandas as pd
import pytest

from skrub import _dataframe as sbd
from skrub import _utils
from skrub._utils import (
    LRUDict,
    import_optional_dependency,
    unique_strings,
)
from skrub.datasets import toy_orders


def test_lrudict():
    dict_ = LRUDict(10)

    for x in range(15):
        dict_[x] = f"filled {x}"

    for x in range(5, 15):
        assert x in dict_
        assert dict_[x] == f"filled {x}"

    for x in range(5):
        assert x not in dict_


@pytest.mark.parametrize(
    "values", [[], [None], ["", None], ["abc", "", None], ["abc", np.nan]]
)
def test_unique_strings(df_module, values):
    if df_module.name == "polars":
        values = [None if pd.isna(v) else v for v in values]
    c = df_module.make_column("", values) if values else df_module.empty_column
    is_null = sbd.to_numpy(sbd.is_null(c))
    unique, idx = unique_strings(sbd.to_numpy(c), is_null)
    assert list(unique) == sorted({s if isinstance(s, str) else "" for s in values})
    assert list(unique[idx]) == [s if isinstance(s, str) else "" for s in values]


def test_import_optional_dependency():
    """Check that we raise the proper error message when an optional dependency is not
    installed."""
    err_msg = "Missing optional dependency 'xxx'.  Use pip or conda to install xxx."
    with pytest.raises(ImportError, match=err_msg):
        import_optional_dependency("xxx")

    err_msg = "Missing optional dependency 'xxx'. Extra text. Use pip or conda"
    with pytest.raises(ImportError, match=err_msg):
        import_optional_dependency("xxx", extra="Extra text.")

    # smoke test for an available dependency
    sklearn_module = import_optional_dependency("sklearn")
    assert ismodule(sklearn_module)


def test_short_repr():
    assert _utils.short_repr({}) == "{}"
    d = {3: 3, 2: 2}
    assert _utils.short_repr(d) == "{3: 3, 2: 2}"
    d = {i: i for i in range(100)}
    assert _utils.short_repr(d) == "{0: 0, 1: 1, 2: 2, 3: 3, ...}"
    d = {}
    for _i in range(10):
        d = {0: d}
    assert _utils.short_repr(d) == "{0: {0: {0: {0: {0: {0: {...}}}}}}}"
    df = toy_orders().X
    assert _utils.short_repr(df) == "DataFrame(...)"

    class A:
        def __skrub_short_repr__(self):
            return "short"

    assert _utils.short_repr(A()) == "short"

    class A:
        def __repr__(self):
            return f"make({list(range(100))})"

    assert _utils.short_repr(A()) == "make([0, 1, 2, 3, 4, 5, 6...)"


def test_passthrough():
    p = _utils.PassThrough()
    X = [1, 2, 3]
    assert p.fit(X) is p
    assert p.fit_transform(X) is X
    assert p.transform(X) is X
    X = [4, 5, 6]
    assert p.transform(X) is X


def test_format_duration():
    assert _utils.format_duration(2 * 3600 + 17 * 60 + 3.5) == "2h 17m 3.5s"
    assert _utils.format_duration(17 * 60 + 3.5) == "0h 17m 3.5s"
    assert _utils.format_duration(3.5) == "0h 0m 3.5s"
    assert _utils.format_duration(3.5279e-5) == "0h 0m 3.5e-05s"
    assert _utils.format_duration(0) == "0h 0m 0s"
    with pytest.raises(ValueError, match=".*only handles non-negative durations"):
        _utils.format_duration(-1)
