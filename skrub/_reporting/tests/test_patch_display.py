import pickle

import pytest

from skrub import config_context, patch_display, set_config, unpatch_display
from skrub.conftest import skip_polars_installed_without_pyarrow


@pytest.mark.parametrize("repeat_patch", [1, 2])
@pytest.mark.parametrize("repeat_unpatch", [1, 2])
@skip_polars_installed_without_pyarrow
def test_patch_display(df_module, repeat_patch, repeat_unpatch, capsys):
    df = df_module.make_dataframe(
        dict(
            a=[1, 2, 3, 4],
            b=["one", "two", "three", "four"],
            c=[11.1, 11.2, 11.3, 11.4],
        )
    )
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

        try:
            capsys.readouterr()
            patch_display(verbose=0)
            df._repr_html_()
            assert capsys.readouterr().err == ""

            capsys.readouterr()
            patch_display()
            df._repr_html_()
            assert capsys.readouterr().err != ""

            capsys.readouterr()
            patch_display(verbose=1)
            df._repr_html_()
            assert capsys.readouterr().err != ""
        finally:
            unpatch_display()


@pytest.mark.parametrize("repeat_patch", [1, 2])
@pytest.mark.parametrize("repeat_unpatch", [1, 2])
@skip_polars_installed_without_pyarrow
def test_patch_display_config(df_module, repeat_patch, repeat_unpatch, capsys):
    df = df_module.make_dataframe(
        dict(
            a=[1, 2, 3, 4],
            b=["one", "two", "three", "four"],
            c=[11.1, 11.2, 11.3, 11.4],
        )
    )
    assert "<table" in df._repr_html_()
    assert "<skrub-table-report" not in df._repr_html_()

    set_config(use_table_report=False)
    assert "<table" in df._repr_html_()
    assert "<skrub-table-report" not in df._repr_html_()
    for _ in range(repeat_patch):
        set_config(use_table_report=True)
        try:
            assert "<skrub-table-report" in df._repr_html_()
            pickle.loads(pickle.dumps(df))
        finally:
            for _ in range(repeat_unpatch):
                set_config(use_table_report=False)
        pickle.loads(pickle.dumps(df))
        assert "<table" in df._repr_html_()
        assert "<skrub-table-report" not in df._repr_html_()

        try:
            capsys.readouterr()
            with config_context(use_table_report=True, table_report_verbosity=0):
                df._repr_html_()
                assert capsys.readouterr().err == ""

            capsys.readouterr()
            with config_context(use_table_report=True, table_report_verbosity=None):
                df._repr_html_()
                assert capsys.readouterr().err != ""

            capsys.readouterr()
            with config_context(use_table_report=True, table_report_verbosity=1):
                df._repr_html_()
                assert capsys.readouterr().err != ""
        finally:
            set_config(use_table_report=False)


def test_max_plot_max_assoc_columns_parameter(pd_module):
    patch_display()
    df = pd_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(10)}
    )
    assert "data-test-plots-skipped" not in df._repr_html_()
    assert "data-test-associations-skipped" not in df._repr_html_()

    df2 = pd_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(30)}
    )
    patch_display()
    assert "data-test-plots-skipped" not in df2._repr_html_()
    assert "data-test-associations-skipped" not in df2._repr_html_()

    df3 = pd_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(31)}
    )
    patch_display()
    assert "data-test-plots-skipped" in df3._repr_html_()
    assert "data-test-associations-skipped" in df3._repr_html_()

    df4 = pd_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(12)}
    )
    patch_display(max_plot_columns=10, max_association_columns=10)
    assert "data-test-plots-skipped" in df4._repr_html_()
    assert "data-test-associations-skipped" in df4._repr_html_()

    df5 = pd_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(12)}
    )
    patch_display(max_plot_columns=15, max_association_columns=15)
    assert "data-test-plots-skipped" not in df5._repr_html_()
    assert "data-test-associations-skipped" not in df5._repr_html_()

    df6 = pd_module.make_dataframe(
        {f"col_{i}": [i + j for j in range(3)] for i in range(5)}
    )
    patch_display(max_plot_columns=None, max_association_columns=None)
    assert "data-test-plots-skipped" not in df6._repr_html_()
    assert "data-test-associations-skipped" not in df6._repr_html_()
