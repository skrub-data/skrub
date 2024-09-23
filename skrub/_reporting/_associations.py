"""Detect which columns have strong statistical associations."""
import warnings

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

from .. import _dataframe as sbd

_N_BINS = 10
_CATEGORICAL_THRESHOLD = 30


def cramer_v(df):
    """Get the Cramer V statistic of association between all pairs of columns.

    The result is returned as a list of tuples (col_1_name, col_2_name,
    statistic_value). As the function is commutative, each pair of columns
    appears only once (either col_1, col_2 or col_2, col_1 but not both).
    The results are sorted from most associated to least associated.

    To compute the statistic, all columns are discretized. Numeric columns are
    binned with 10 bins. For categorical columns, only the 10 most frequent
    categories are considered. In both cases, nulls are treated as a separate
    category, ie a separate row in the contingency table. Thus associations
    betwen the values of 2 columns or between their missingness patterns may be
    captured.

    Parameters
    ----------
    df : dataframe
        The dataframe whose columns will be compared to each other.

    Returns
    -------
    list of (str, str, float) tuples
        The associations: each tuple contains the names of the two columns and
        the corresponding V statistic.
    """
    return _stack_symmetric_associations(_cramer_v_matrix(df), sbd.column_names(df))


def _stack_symmetric_associations(associations, column_names):
    """Turn a symmetric matrix of V statistics into a list of (col, col, value).

    The input is the symmetric matrix where entry i, j contains the V statistic
    for (column i, column j).

    The result is a list of (column i name, column j name, V statistic). Each
    pair of column appears only once ie the number of entries is
    n_columns x (n_columns - 1). The results are sorted from most to least
    associated.
    """
    left_indices, right_indices = np.triu_indices_from(associations, 1)
    associations = associations[(left_indices, right_indices)]
    order = np.argsort(associations)[::-1]
    left_indices, right_indices, associations = (
        left_indices[order],
        right_indices[order],
        associations[order],
    )
    return [
        {
            "left_column_name": column_names[left],
            "left_column_idx": int(left),
            "right_column_name": column_names[right],
            "right_column_idx": int(right),
            "cramer_v": float(a),
        }
        for (left, right, a) in zip(left_indices, right_indices, associations)
    ]


def _cramer_v_matrix(df):
    """Compute Cramer's V statistic for all pairs of columns.

    The result is a symmetric matrix where entry (i, j) contains the V
    statistic of association between column i and column j.
    """
    encoded = _onehot_encode(df, _N_BINS)
    table = _contingency_table(encoded)
    stats = _compute_cramer(table, sbd.shape(df)[0])
    return stats


def _onehot_encode(df, n_bins):
    """One-hot encode all columns in a dataframe.

    Numeric columns with fewer than _CATEGORICAL_THRESHOLD values are treated
    as categorical, the others are binned with uniform bins before being
    one-hot encoded. For categorical columns infrequent categories are lumped
    together.

    Returns an array of shape (n columns, n bins, n rows) where result[i]
    contains the one-hot-encoding representation of column i.
    """
    n_rows, n_cols = sbd.shape(df)
    output = np.zeros((n_cols, n_bins, n_rows), dtype=bool)
    for col_idx, col_name in enumerate(sbd.column_names(df)):
        col = sbd.col(df, col_name)
        if sbd.is_numeric(col):
            col = sbd.to_float32(col)
            if _CATEGORICAL_THRESHOLD <= sbd.n_unique(col):
                _onehot_encode_numbers(sbd.to_numpy(col), n_bins, output[col_idx])
            else:
                _onehot_encode_categories(sbd.to_numpy(col), n_bins, output[col_idx])
        else:
            col = sbd.to_string(col)
            _onehot_encode_categories(sbd.to_numpy(col), n_bins, output[col_idx])
    return output


def _onehot_encode_categories(values, n_bins, output):
    """One-hot encode a categorical column."""
    encoded = OneHotEncoder(max_categories=n_bins, sparse_output=False).fit_transform(
        values[:, None]
    )
    effective_n_bins = encoded.shape[1]
    output[:effective_n_bins] = encoded.T


def _onehot_encode_numbers(values, n_bins, output):
    """One-hot encode a numeric column."""
    mask = ~np.isfinite(values)
    filled_na = np.array(values)
    # TODO pick a better value & non-uniform bins?
    filled_na[mask] = 0.0
    encoder = KBinsDiscretizer(
        n_bins=n_bins - 1,
        strategy="uniform",
        subsample=None,
        encode="onehot-dense",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        encoded = encoder.fit_transform(filled_na[:, None])
    encoded[mask] = 0
    effective_n_bins = encoded.shape[1]
    output[:effective_n_bins] = encoded.T
    output[effective_n_bins] = mask


def _contingency_table(encoded):
    """Build the contingency table given a OH-encoded dataframe.

    The input is computed by ``_one_hot_encode``:
    it has shape (n columns, n bins, n rows).

    This function computes for each pair of columns, the n bins x n bins
    contingency table that will be used to measure their association.

    The result is an array of shape n cols, n cols, n bins, n bind where result
    [i, j, :, :] is the contingency table for column i vs column j.
    """
    n_cols, n_bins, _ = encoded.shape
    out = np.empty((n_cols, n_cols, n_bins, n_bins), dtype="int32")
    return np.einsum("ack,bdk", encoded, encoded, out=out)


def _compute_cramer(table, n_samples):
    """Compute the Cramer V statistic given a contingency table.

    The input is the table computed by ``_contingency_table`` with shape
    (n cols, n cols, n bins, n bins).

    This returs the symmetric matrix with shape (n cols, n cols) where entry
    i, j contains the statistic for column i x column j.
    """
    marginal_0 = table.sum(axis=-2)
    marginal_1 = table.sum(axis=-1)
    expected = (
        marginal_0[:, :, None, :]
        * marginal_1[:, :, :, None]
        / marginal_0.sum(axis=-1)[:, :, None, None]
    )
    diff = table - expected
    expected[expected == 0] = 1
    chi_stat = ((diff**2) / expected).sum(axis=-1).sum(axis=-1)
    min_dim = np.minimum(
        (marginal_0 > 0).sum(axis=-1) - 1, (marginal_1 > 0).sum(axis=-1) - 1
    )
    stat = np.sqrt(chi_stat / (n_samples * np.maximum(min_dim, 1)))
    stat[min_dim == 0] = 0.0
    return stat
