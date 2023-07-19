from inspect import ismodule

import pytest

from skrub._utils import LRUDict, import_optional_dependency


def test_lrudict():
    dict_ = LRUDict(10)

    for x in range(15):
        dict_[x] = f"filled {x}"

    for x in range(5, 15):
        assert x in dict_
        assert dict_[x] == f"filled {x}"

    for x in range(5):
        assert x not in dict_


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
