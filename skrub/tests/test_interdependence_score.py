import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal

# from skrub import _dataframe as sbd
from skrub._interdependence_score import _ids_matrix


@pytest.fixture(scope="function")
def variables():
    np.random.seed(42)
    n = 1000

    f0 = np.random.randn(n)
    f1 = 0.8 * f0 + 0.2 * np.random.randn(n)
    f2 = np.random.randn(n)
    f3 = np.sin(f2)
    f4 = np.square(f2)
    f5 = np.random.randn(n)
    f6 = np.tanh(f0 * f2)
    f7 = pd.qcut(
        f1, q=5, labels=["cat1_0", "cat1_1", "cat1_2", "cat1_3", "cat1_4"]
    ).to_numpy()
    f8 = pd.qcut(
        np.random.randn(n), q=3, labels=["cat2_0", "cat2_1", "cat2_2"]
    ).to_numpy()

    X_num = np.column_stack([f0, f1, f2, f3, f4, f5, f6])
    X_cat = np.column_stack((f7, f8))

    X_dict = {f"Num-{i}": X_num[:, i] for i in range(X_num.shape[1])}
    X_dict.update({f"Cat-{i+1}": X_cat[:, i] for i in range(X_cat.shape[1])})

    return X_dict


REFERENCE_SCORES = np.array(
    [
        [1.0, 0.94, 0.05, 0.06, 0.08, 0.07, 0.63, 0.7, 0.04],
        [0.94, 1.0, 0.06, 0.05, 0.07, 0.07, 0.56, 0.73, 0.05],
        [0.05, 0.06, 1.0, 0.99, 0.98, 0.07, 0.62, 0.07, 0.07],
        [0.06, 0.05, 0.99, 1.0, 0.89, 0.07, 0.59, 0.06, 0.07],
        [0.08, 0.07, 0.98, 0.89, 1.0, 0.06, 0.59, 0.08, 0.07],
        [0.07, 0.07, 0.07, 0.07, 0.06, 1.0, 0.05, 0.08, 0.06],
        [0.63, 0.56, 0.62, 0.59, 0.59, 0.05, 1.0, 0.34, 0.05],
        [0.7, 0.73, 0.07, 0.06, 0.08, 0.08, 0.34, 1.0, 0.05],
        [0.04, 0.05, 0.07, 0.07, 0.07, 0.06, 0.05, 0.05, 1.0],
    ]
)


def test_interdependence_score(df_module, variables):
    df = df_module.make_dataframe(variables)
    interdependence_scores, pvalues = _ids_matrix(df, p_val=True)
    n_variables = len(variables)

    assert_almost_equal(np.diag(interdependence_scores), np.ones(n_variables))
    diff_mat = np.abs(REFERENCE_SCORES - interdependence_scores)
    assert np.all(diff_mat < 0.05)
    assert np.all(pvalues[interdependence_scores > 0.5] < 0.05)
