import builtins
import sys
import warnings

import numpy as np
import pytest

from skrub import _dataframe as sbd
from skrub import column_associations
from skrub.conftest import skip_polars_installed_without_pyarrow


@skip_polars_installed_without_pyarrow
def test_column_associations(df_module):
    x = (np.ones((7, 3)) * np.arange(3)).ravel()
    # y is a monotonic but non-linear function of x
    y = 2 - 3 * x**2
    z = np.arange(len(x))
    df = df_module.make_dataframe(dict(x=x, y=y, z=z))
    asso = column_associations(df)
    asso = sbd.pandas_convert_dtypes(asso)
    assert sbd.to_list(sbd.col(asso, "left_column_name")) == ["x", "y", "x"]
    assert sbd.to_list(sbd.col(asso, "left_column_idx")) == [0, 1, 0]
    assert sbd.to_list(sbd.col(asso, "right_column_name")) == ["y", "z", "z"]
    assert sbd.to_list(sbd.col(asso, "right_column_idx")) == [1, 2, 2]
    assert sbd.to_list(sbd.col(asso, "cramer_v")) == pytest.approx(
        [1.0, 0.6546536, 0.6546536]
    )
    pearson = sbd.to_list(sbd.col(asso, "pearson_corr"))
    spearman = sbd.to_list(sbd.col(asso, "spearman_corr"))
    assert pearson == pytest.approx([-0.9607689, -0.1295501, 0.1348400])
    assert spearman == pytest.approx([-1.0, -0.1348400, 0.1348400])
    # Spearman only depends on the ranks, so it differs from Pearson for the
    # pairs involving the non-linear y, but not for the (x, z) pair.
    assert pearson[0] != pytest.approx(spearman[0])
    assert pearson[1] != pytest.approx(spearman[1])
    assert pearson[2] == pytest.approx(spearman[2])
    assert "kendall_corr" not in sbd.column_names(asso)


@skip_polars_installed_without_pyarrow
def test_column_associations_with_kendall(df_module):
    x = (np.ones((7, 3)) * np.arange(3)).ravel()
    y = 2 - 3 * x
    z = np.arange(len(x))
    df = df_module.make_dataframe(dict(x=x, y=y, z=z))
    asso = column_associations(df, compute_kendall=True)
    asso = sbd.pandas_convert_dtypes(asso)
    assert sbd.to_list(sbd.col(asso, "left_column_name")) == ["x", "y", "x"]
    assert sbd.to_list(sbd.col(asso, "right_column_name")) == ["y", "z", "z"]
    assert sbd.to_list(sbd.col(asso, "kendall_corr")) == pytest.approx(
        [-1.0, -0.1195229, 0.1195229]
    )


@skip_polars_installed_without_pyarrow
@pytest.mark.parametrize(
    "params, expected_corr_columns",
    [
        ({}, ["pearson_corr", "spearman_corr"]),
        ({"compute_pearson": False}, ["spearman_corr"]),
        ({"compute_spearman": False}, ["pearson_corr"]),
        (
            {"compute_kendall": True},
            ["pearson_corr", "spearman_corr", "kendall_corr"],
        ),
        (
            {"compute_pearson": False, "compute_spearman": False},
            [],
        ),
        (
            {
                "compute_pearson": False,
                "compute_spearman": False,
                "compute_kendall": True,
            },
            ["kendall_corr"],
        ),
    ],
)
def test_column_associations_correlation_columns(
    df_module, params, expected_corr_columns
):
    df = df_module.make_dataframe({"x": [0, 1, 2], "y": [0, 1, 3]})

    associations = column_associations(df, **params)

    assert sbd.column_names(associations) == [
        "left_column_name",
        "left_column_idx",
        "right_column_name",
        "right_column_idx",
        "cramer_v",
        *expected_corr_columns,
    ]


def test_column_associations_without_correlations(df_module):
    # Without any correlation, PyArrow is not needed for polars
    df = df_module.make_dataframe({"x": [0, 1], "y": [0, 1]})

    associations = column_associations(
        df, compute_pearson=False, compute_spearman=False
    )

    assert sbd.column_names(associations) == [
        "left_column_name",
        "left_column_idx",
        "right_column_name",
        "right_column_idx",
        "cramer_v",
    ]


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"compute_pearson": False},
        {"compute_pearson": False, "compute_spearman": False, "compute_kendall": True},
    ],
)
def test_column_associations_polars_without_pyarrow_raises(monkeypatch, params):
    """Requesting any correlation without PyArrow raises an exception."""
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"x": [0, 1], "y": [0, 1]})
    monkeypatch.delitem(sys.modules, "pyarrow", raising=False)
    builtin_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError(name)
        return builtin_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)

    with pytest.raises(ImportError):
        column_associations(df, **params)


@skip_polars_installed_without_pyarrow
def test_infinite(df_module):
    # non-regression test for https://github.com/skrub-data/skrub/issues/1133
    # (column associations would raise an exception on low-cardinality float
    # column with infinite values)
    with warnings.catch_warnings():
        # pandas convert_dtypes() emits a spurious warning while trying to decide if
        # floats should be cast to int or not
        # eg `pd.Series([float('inf')]).convert_dtypes()` raises the warning
        warnings.filterwarnings("ignore", message="invalid value encountered in cast")

        column_associations(
            df_module.make_dataframe({"a": [float("inf"), 1.5], "b": [0.0, 1.5]})
        )
