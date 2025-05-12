import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import skrub._dataframe as sbd
from skrub import GapEncoder, StringEncoder
from skrub._total_std_scaler import total_standard_deviation_scaler

try:
    import sentence_transformers  # noqa: F401

    TRANSFORMERS = True
except ImportError:
    TRANSFORMERS = False


def test_scaling_invariance():
    """Normalizing twice leads to a unit scale."""
    X = np.random.randn(10, 3)
    scale_1 = total_standard_deviation_scaler(X)
    X_scale_1 = X / scale_1

    scale_2 = total_standard_deviation_scaler(X_scale_1)
    X_scale_2 = X_scale_1 / scale_2

    assert_array_almost_equal(X_scale_1, X_scale_2)

    assert scale_2 == pytest.approx(1)


def test_nonfinite():
    """Nonfinite values are equivalent to removing these value
    column-wise.
    """
    X = np.array([[np.nan, np.nan, np.nan], [1, 2, 3], [4, 5, 6]])
    assert total_standard_deviation_scaler(X) == total_standard_deviation_scaler(X[1:])


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
    assert total_standard_deviation_scaler(np.array(X_t)) == pytest.approx(1)


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
    assert total_standard_deviation_scaler(Xt_np) == pytest.approx(1)
