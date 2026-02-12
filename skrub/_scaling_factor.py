"""
Skrub encoders output vectors that don't have the same scale. Scaling using the
standard deviation of the matrix help computational stability and downstream prediction
performance.
"""

import numpy as np


def _clip_epsilon(scaling_factor):
    # Avoid division by very small or zero values.
    if scaling_factor < 10 * np.finfo(scaling_factor.dtype).eps:
        scaling_factor = 1
    return scaling_factor


def scaling_factor(X):
    r"""Compute the total standard deviation scaler of X.

    This scaling factor is used to normalize the vectors outputs of
    :class:`StringEncoder`, :class:`TextEncoder` and :class:`GapEncoder`. It is computed
    during ``fit`` and applied during ``transform``.

    Conceptually, this coefficient corresponds to the square root of the sum of the
    variances of all columns of X.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        The matrix to compute the norm from.

    Returns
    -------
    scaling_factor : float,
        The scaling factor.

    See Also
    --------
    :class:`~sklearn.preprocessing.Normalizer` :
        Performs row-wise normalization.

    Notes
    -----
    We define the scaler as:

    .. math::

        ||X||_B = \sqrt{\sum_{j=1}^D \sigma^2(X_j)} \in \mathbb{R}

    where:

    * :math:`\sigma^2(X_j)` is the population variance of the column :math:`j`
    * :math:`D` is the number of features

    When :math:`D = 1`, this scaler is the population standard deviation of the column.

    We then rescale every element of :math:`X` using this scaler.
    """
    scaling_factor = np.sqrt(np.nansum(np.nanvar(X, ddof=0, axis=0)))

    return _clip_epsilon(scaling_factor)
