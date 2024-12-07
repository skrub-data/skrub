import pickle

import pytest

from skrub import patch_display, unpatch_display


@pytest.mark.parametrize("repeat_patch", [1, 2])
@pytest.mark.parametrize("repeat_unpatch", [1, 2])
def test_patch_display(df_module, repeat_patch, repeat_unpatch, capsys):
    df = df_module.example_dataframe
    assert "<table" in df._repr_html_()
    assert "<skrub-table-report" not in df._repr_html_()
    patch_display(pandas=False, polars=False)
    assert "<table" in df._repr_html_()
    assert "<skrub-table-report" not in df._repr_html_()
    for _ in range(repeat_patch):
        patch_display()
    try:
        assert "<skrub-table-report" in df._repr_html_()
        pickle.loads(pickle.dumps(df))
    finally:
        for _ in range(repeat_unpatch):
            unpatch_display()
    pickle.loads(pickle.dumps(df))
    assert "<table" in df._repr_html_()
    assert "<skrub-table-report" not in df._repr_html_()

    unpatch_display()
    try:
        capsys.readouterr()
        patch_display(verbose=0)
        df._repr_html_()
        assert capsys.readouterr().out == ""

        capsys.readouterr()
        patch_display()
        df._repr_html_()
        assert capsys.readouterr().out != ""

        capsys.readouterr()
        patch_display(verbose=1)
        df._repr_html_()
        assert capsys.readouterr().out != ""
    finally:
        unpatch_display()
