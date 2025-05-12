import numpy as np


def total_standard_deviation_scaler(X):
    r"""Compute the total standard deviation scaler of X.

    This scaler is used to normalize the vectors outputs of :class:`StringEncoder`,
    :class:`TextEncoder` and :class:`GapEncoder`. It is computed during ``fit`` and
    applied during ``transform``.

    Conceptually, this scaler corresponds to the square root of the sum of the variances
    of all columns of X.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        The matrix to compute the norm from.

    Returns
    -------
    scaler : float,
        The total standard deviation scaler.

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
    norm = np.sqrt(np.nansum(np.nanvar(X, ddof=0, axis=0)))

    # Avoid division by very small or zero values.
    if norm < 10 * np.finfo(norm.dtype).eps:
        norm = 1

    return norm


def batch_standard_deviation_scaler(X, past_stats):
    """Compute the batched version of the total standard deviation scaler.

    This function is used to compute the scaler in ``GapEncoder.partial_fit``, when
    the dataset is only accessed via batches coming one by one. This function
    accumulates statistics obtained from previous partial fit to compute the scaler.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        The batch to update the norm with.

    past_stats : defaultdict(Counter)
        Past statistics container, where columns map their combined statistics.

    Returns
    -------
    scaler : float
        The total standard deviation scaler, up to the last batch

    past_stats : defaultdict(Counter)
        Past statistics container, updated with the last batch.
    """
    total_combined_var = 0

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

        total_combined_var += combined_var
        past_stats[j] = {
            "n": N,
            "mean": combined_mean,
            "var": combined_var,
        }

    return np.sqrt(total_combined_var), past_stats
