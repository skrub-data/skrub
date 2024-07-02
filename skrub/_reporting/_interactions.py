import warnings

import numpy as np
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer

from .. import _dataframe as sbd
from .._to_str import ToStr

_N_BINS = 10
_CATEGORICAL_THRESHOLD = 30


def stack_symmetric_associations(associations, column_names):
    left_indices, right_indices = np.triu_indices_from(associations, 1)
    associations = associations[(left_indices, right_indices)]
    order = np.argsort(associations)[::-1]
    left_indices, right_indices, associations = (
        left_indices[order],
        right_indices[order],
        associations[order],
    )
    return [
        (column_names[left], column_names[right], a)
        for (left, right, a) in zip(left_indices, right_indices, associations)
    ]


def cramer_v(df):
    encoded = _onehot_encode(df, _N_BINS)
    table = _contingency_table(encoded)
    stats = _compute_cramer(table, sbd.shape(df)[0])
    return stats


def _onehot_encode(df, n_bins):
    n_rows, n_cols = sbd.shape(df)
    output = np.zeros((n_cols, n_bins, n_rows), dtype=bool)
    for col_idx, col_name in enumerate(sbd.column_names(df)):
        col = sbd.col(df, col_name)
        if sbd.is_numeric(col):
            _onehot_encode_numbers(sbd.to_numpy(col), n_bins, output[col_idx])
        else:
            col = ToStr().fit_transform(col)
            _onehot_encode_categories(sbd.to_numpy(col), n_bins, output[col_idx])
    return output


def _onehot_encode_categories(values, n_bins, output):
    encoded = OneHotEncoder(max_categories=n_bins, sparse_output=False).fit_transform(
        values[:, None]
    )
    effective_n_bins = encoded.shape[1]
    output[:effective_n_bins] = encoded.T


def _onehot_encode_numbers(values, n_bins, output):
    values = values.astype(float)
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
    n_cols, n_quantiles, _ = encoded.shape
    out = np.empty((n_cols, n_cols, n_quantiles, n_quantiles), dtype="int32")
    return np.einsum("ack,bdk", encoded, encoded, out=out)


def _compute_cramer(table, n_samples):
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
