import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import skrub
import skrub._dataframe as sbd
from skrub import GapEncoder, StringEncoder
from skrub._scaling_factor import scaling_factor

try:
    import sentence_transformers  # noqa: F401

    TRANSFORMERS = True
except ImportError:
    TRANSFORMERS = False


def test_scaling_invariance():
    """Normalizing twice leads to a unit scale."""
    X = np.random.randn(10, 3)
    scale_1 = scaling_factor(X)
    X_scale_1 = X / scale_1

    scale_2 = scaling_factor(X_scale_1)
    X_scale_2 = X_scale_1 / scale_2

    assert_array_almost_equal(X_scale_1, X_scale_2)

    assert scale_2 == pytest.approx(1)


def test_nonfinite():
    """Nonfinite values are equivalent to removing these value
    column-wise.
    """
    X = np.array([[np.nan, np.nan, np.nan], [1, 2, 3], [4, 5, 6]])
    assert scaling_factor(X) == scaling_factor(X[1:])


@pytest.mark.parametrize(
    "encoder",
    [
        StringEncoder(n_components=2),
        GapEncoder(n_components=2),
        pytest.param(
            skrub.TextEncoder(
                model_name="sentence-transformers/paraphrase-albert-small-v2",
                n_components=2,
                device="cpu",
            ),
            marks=pytest.mark.skipif(
                not TRANSFORMERS, reason="transformers not installed"
            ),
        ),
    ],
)
def test_encoders(df_module, encoder):
    X = df_module.example_dataframe["str-col"]
    X_t = encoder.fit_transform(X)
    assert scaling_factor(np.array(X_t)) == pytest.approx(1, abs=0.1)


def test_partial_fit(df_module):
    X = df_module.example_dataframe["str-col"]
    X1 = sbd.slice(X, 0, 2)
    X2 = sbd.slice(X, 2, 4)

    gap = GapEncoder(n_components=2)
    gap.partial_fit(X1)
    gap.partial_fit(X2)

    assert gap.scaling_factor_ == pytest.approx(0.75)
