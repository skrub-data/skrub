import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from skrub import BlockNormalizerL2, GapEncoder, StringEncoder
from skrub._block_normalizer import _avg_norm


def test_normalization_invariance():
    """Normalizing twice leads to a unit norm."""
    X = np.random.randn(10, 3)
    normalizer = BlockNormalizerL2()
    X_t = normalizer.fit_transform(X)
    X_t2 = normalizer.fit_transform(X_t)

    assert_array_almost_equal(X_t, X_t2)
    assert normalizer.avg_norm_ == pytest.approx(1)


def test_nonfinite():
    """Nonfinite values are equivalent to removing these value
    column-wise.
    """
    X = np.array([[None, np.inf, np.nan], [1, 2, 3], [4, 5, 6]])

    normalizer = BlockNormalizerL2()
    X_t = normalizer.fit_transform(X)

    normalizer_truncated = BlockNormalizerL2()
    X_trunc_t = normalizer_truncated.fit_transform(X[1:])

    assert_array_equal(X_t[1:], X_trunc_t)
    assert normalizer.avg_norm_ == normalizer_truncated.avg_norm_


def test_dataframe(df_module):
    normalizer = BlockNormalizerL2()
    X = df_module.example_dataframe[
        ["int-col", "int-not-null-col", "float-col", "bool-col", "bool-not-null-col"]
    ]
    X_t = normalizer.fit_transform(X)
    assert _avg_norm(X_t) == pytest.approx(1)


@pytest.mark.parametrize(
    "encoder",
    [
        StringEncoder(block_normalize=True, n_components=2),
        GapEncoder(block_normalize=True, n_components=2),
    ],
)
def test_encoders(df_module, encoder):
    X = df_module.example_dataframe["str-col"]
    X_t = encoder.fit_transform(X)
    assert _avg_norm(np.array(X_t)) == pytest.approx(1)


def test_non_num_values(df_module):
    with pytest.raises(ValueError, match="only accept numeric columns"):
        BlockNormalizerL2().fit_transform(df_module.example_dataframe)

    with pytest.raises(ValueError, match="only accept numeric values"):
        BlockNormalizerL2().fit_transform(np.array([[1, "a"]]))
