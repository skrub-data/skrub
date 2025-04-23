import numpy as np
import sklearn
from sklearn.utils.fixes import parse_version
from sklearn.utils.validation import check_array

from . import _dataframe as sbd


def total_variance_norm(X):
    r"""Fit the average L2 norm and apply the normalization.

    Compute the average L2 norm (a scalar) from a numerical DataFrame or a 2D NumPy
    array, and use this norm to normalize the input element-wise. This computation
    is robust to non-finite values such as ``np.inf`` or ``np.nan``, but raises an
    error if string values are present.

    See Also
    --------
    :class:`~sklearn.preprocessing.Normalizer` :
        Performs row-wise normalization.

    Notes
    -----
    We define this norm as:

    .. math::

        ||X||_B = \sqrt{\sum_{j=1}^D \sigma^2(X_j)} \in \mathbb{R}

    where:

    * :math:`\sigma^2(X_j)` is the population variance of the column :math:`j`
    * :math:`D` is the number of features

    When :math:`D = 1`, this norm is the population standard deviation of the column.
    We then normalize each element as:

    .. math::

        \forall i,j, \quad \tilde{X}_{ij} = \frac{X_{ij}}{||X||_B}
    """

    if hasattr(X, "__dataframe__"):
        msg = "X must only have numeric columns, but {col} is {dtype}."
        for col in sbd.to_column_list(X):
            if not (sbd.is_numeric(col) or sbd.is_bool(col)):
                raise ValueError(msg.format(col=col.name, dtype=sbd.dtype(col)))
    elif isinstance(X, np.ndarray):
        msg = "X must only have numeric values, but at least one value is non-numeric."
        try:
            X.astype("float32")
        except ValueError as e:
            raise ValueError(msg) from e
    else:
        raise TypeError(
            f"X must be a Pandas, Polars dataframe or a Numpy array, got {type(X)}."
        )

    sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)
    all_finite_key = (
        "force_all_finite"
        if sklearn_version < parse_version("1.6")
        else "ensure_all_finite"
    )
    X = check_array(
        X,
        accept_sparse=False,
        dtype="numeric",
        ensure_2d=True,
        copy=False,
        **{all_finite_key: False},
    )

    # Sanity check: replace inf values with nan
    mask_finite = np.isfinite(X)
    X[~mask_finite] = np.nan

    norm = np.sqrt(np.nanvar(X, ddof=0, axis=0).sum())

    # Avoid division by very small or zero values.
    if norm < 10 * np.finfo(norm.dtype).eps:
        norm = 1

    return norm


def accumulate_norm(X, mean1, var1, n1):
    total_combined_var = 0
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n2 = X.shape[0]
    for j in range(X.shape[1]):
        x_col = X[:, j]
        mean2, var2 = x_col.mean(), x_col.var(ddof=0)

        delta = mean2 - mean1

        total_combined_var += (
            n1 * var1 + n2 * var2 + (n1 * n2 * delta**2) / (n1 + n2)
        ) / (n1 + n2)

    return (np.sqrt(total_combined_var), {"mean1": mean2, "var1": var2, "n1": n2})
