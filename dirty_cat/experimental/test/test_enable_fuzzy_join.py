"""Tests for making sure experimental import of fuzzy_join work as expected."""

import textwrap

from sklearn.utils._testing import assert_run_python_script


def test_import_fuzzy_join():
    good_import = """
    from dirty_cat.experimental import enable_fuzzy_join
    from dirty_cat import fuzzy_join
    """
    assert_run_python_script(textwrap.dedent(good_import))

    bad_imports = """
    import pytest
    with pytest.raises(ImportError):
        from dirty_cat import fuzzy_join
    """
    assert_run_python_script(textwrap.dedent(bad_imports))
