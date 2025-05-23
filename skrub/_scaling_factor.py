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


def scaling_factor_batch(X, past_stats):
    """Compute the batched version of the scaling factor.

    This function is used to compute the scaling factor in ``GapEncoder.partial_fit``,
    when the dataset is only accessed via batches coming one by one. This function
    accumulates statistics obtained from previous partial fit to compute the
    scaling factor.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        The batch to update the scaling factor with.

    past_stats : defaultdict(Counter)
        Past statistics container, where columns map their combined statistics.

    Returns
    -------
    scaling_factor : float
        The total standard deviation scaler, up to the last batch

    past_stats : defaultdict(Counter)
        Past statistics container, updated with the last batch.
    """
    scaling_factor_sq = 0

    n2 = X.shape[0]
    for j in range(X.shape[1]):
        x_col = X[:, j]
        mean2, var2 = x_col.mean(), x_col.var(ddof=0)
        n1, mean1, var1 = (
            past_stats[j]["n"],
            past_stats[j]["mean"],
            past_stats[j]["var"],
        )
        N = n1 + n2
        delta = mean2 - mean1

        combined_mean = (mean1 * n1 + mean2 * n2) / N
        combined_var = (n1 * var1 + n2 * var2 + (n1 * n2 * delta**2) / N) / N

        scaling_factor_sq += combined_var
        past_stats[j] = {
            "n": N,
            "mean": combined_mean,
            "var": combined_var,
        }

    return _clip_epsilon(np.sqrt(scaling_factor_sq)), past_stats
