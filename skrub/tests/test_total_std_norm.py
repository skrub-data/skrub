import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import skrub._dataframe as sbd
from skrub import GapEncoder, StringEncoder, TextEncoder
from skrub._total_std_norm import total_standard_deviation_norm

try:
    import sentence_transformers  # noqa: F401

    TRANSFORMERS = True
except ImportError:
    TRANSFORMERS = False


def test_normalization_invariance():
    """Normalizing twice leads to a unit norm."""
    X = np.random.randn(10, 3)
    norm_1 = total_standard_deviation_norm(X)
    X_norm_1 = X / norm_1

    norm_2 = total_standard_deviation_norm(X_norm_1)
    X_norm_2 = X_norm_1 / norm_2

    assert_array_almost_equal(X_norm_1, X_norm_2)

    assert norm_2 == pytest.approx(1)


def test_nonfinite():
    """Nonfinite values are equivalent to removing these value
    column-wise.
    """
    X = np.array([[np.nan, np.nan, np.nan], [1, 2, 3], [4, 5, 6]])
    assert total_standard_deviation_norm(X) == total_standard_deviation_norm(X[1:])


@pytest.mark.parametrize(
    "encoder",
    [
        StringEncoder(n_components=2),
        GapEncoder(n_components=2),
        pytest.param(
            TextEncoder(n_components=2),
            marks=pytest.mark.skipif(
                not TRANSFORMERS, reason="missing sentence-transformers"
            ),
        ),
    ],
)
def test_encoders(df_module, encoder):
    X = df_module.example_dataframe["str-col"]
    X_t = encoder.fit_transform(X)
    assert total_standard_deviation_norm(np.array(X_t)) == pytest.approx(1)


def test_partial_fit(df_module):
    X = df_module.example_dataframe["str-col"]
    X1 = sbd.slice(X, 0, 2)
    X2 = sbd.slice(X, 2, 4)
    X_full = sbd.slice(X, 0, 4)

    gap = GapEncoder(n_components=2)
    gap.partial_fit(X1)
    gap.partial_fit(X2)
    Xt = gap.transform(X_full)

    Xt_np = np.hstack(
        [sbd.to_numpy(col).reshape(-1, 1) for col in sbd.to_column_list(Xt)]
    )
    assert total_standard_deviation_norm(Xt_np) == pytest.approx(1)
