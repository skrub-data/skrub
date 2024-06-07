from inspect import ismodule

import numpy as np
import pytest

from skrub import _dataframe as sbd
from skrub._utils import LRUDict, import_optional_dependency, unique_strings


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
    "values", [[], [None], ["", None], ["abc", "", None], [np.nan, "abc"]]
)
def test_unique_strings(df_module, values):
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
