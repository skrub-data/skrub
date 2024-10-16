from skrub import patch_display, unpatch_display


def test_patch_display(df_module):
    df = df_module.example_dataframe
    assert "<table" in df._repr_html_()
    assert "<skrub-table-report" not in df._repr_html_()
    patch_display(pandas=False, polars=False)
    assert "<table" in df._repr_html_()
    assert "<skrub-table-report" not in df._repr_html_()
    patch_display()
    try:
        assert "<skrub-table-report" in df._repr_html_()
    finally:
        unpatch_display()
    assert "<table" in df._repr_html_()
    assert "<skrub-table-report" not in df._repr_html_()
