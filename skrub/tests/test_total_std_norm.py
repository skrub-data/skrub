import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import skrub._dataframe as sbd
from skrub import GapEncoder, StringEncoder
from skrub._total_std_norm import total_std_norm

try:
    import sentence_transformers  # noqa: F401

    TRANSFORMERS = True
except ImportError:
    TRANSFORMERS = False


def test_normalization_invariance():
    """Normalizing twice leads to a unit norm."""
    X = np.random.randn(10, 3)
    norm_1 = total_std_norm(X)
    X_norm_1 = X / norm_1

    norm_2 = total_std_norm(X_norm_1)
    X_norm_2 = X_norm_1 / norm_2

    assert_array_almost_equal(X_norm_1, X_norm_2)

    assert norm_2 == pytest.approx(1)


def test_nonfinite():
    """Nonfinite values are equivalent to removing these value
    column-wise.
    """
    X = np.array([[None, np.inf, np.nan], [1, 2, 3], [4, 5, 6]])
    assert total_std_norm(X) == total_std_norm(X[1:])


def test_dataframe(df_module):
    X = df_module.example_dataframe[
        ["int-col", "int-not-null-col", "float-col", "bool-col", "bool-not-null-col"]
    ]
    norm = total_std_norm(X)
    assert total_std_norm(X / norm) == pytest.approx(1)


@pytest.mark.parametrize(
    "encoder",
    [
        StringEncoder(n_components=2),
        GapEncoder(n_components=2),
    ],
)
def test_encoders(df_module, encoder):
    X = df_module.example_dataframe["str-col"]
    X_t = encoder.fit_transform(X)
    assert total_std_norm(np.array(X_t)) == pytest.approx(1)


def test_non_num_values(df_module):
    with pytest.raises(ValueError, match="must only have numeric columns"):
        total_std_norm(df_module.example_dataframe)

    with pytest.raises(ValueError, match="must only have numeric values"):
        total_std_norm(np.array([[1, "a"]]))

    with pytest.raises(
        TypeError, match="must be a Pandas, Polars dataframe or a Numpy"
    ):
        total_std_norm([1, 3])


def test_partial_fit(df_module):
    X = df_module.example_dataframe["str-col"]
    X1 = sbd.slice(X, 0, 2)
    X2 = sbd.slice(X, 2, 4)
    X_full = sbd.slice(X, 0, 4)

    gap = GapEncoder(n_components=2)
    gap.partial_fit(X1)
    gap.partial_fit(X2)
    Xt = gap.transform(X_full)

    assert total_std_norm(Xt) == pytest.approx(1)
