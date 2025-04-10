"""Detect which columns have strong statistical associations."""

import warnings

import numpy as np
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder
from sklearn.utils.fixes import parse_version

from . import _dataframe as sbd
from . import _join_utils

_N_BINS = 10
_CATEGORICAL_THRESHOLD = 30


def column_associations(df):
    """Get measures of statistical associations between all pairs of columns.

    Reported metrics include Cramer's V statistic and Pearson's Correlation
    Coefficient. More may be added in the future.

    The result is returned as a dataframe with columns:

    ``['left_column_name', 'left_column_idx', 'right_column_name',
    'right_column_idx', 'cramer_v', 'pearson_corr']``

    As the function is commutative, each pair of columns appears only once
    (either ``col_1``, ``col_2`` or ``col_2``, ``col_1`` but not both).
    The results are sorted from most associated to least associated.

    To compute the Cramer's V statistic, all columns are discretized. Numeric
    columns are binned with 10 bins. For categorical columns, only the 10 most
    frequent categories are considered. In both cases, nulls are treated as a
    separate category, ie a separate row in the contingency table. Thus
    associations between the values of 2 columns or between their missingness
    patterns may be captured.

    To compute the Pearson's Correlation Coefficient, only numeric columns are
    considered. The correlation is computed using the Pearson method used in
    pandas or polars, depending on the dataframe. In both case, lines containing NaNs
    are dropped

    Parameters
    ----------
    df : dataframe
        The dataframe whose columns will be compared to each other.

    Returns
    -------
    dataframe
        The computed associations.

    Notes
    -----
    Cramér's V is a measure of association between two nominal variables,
    giving a value between 0 and +1 (inclusive).

    * `Cramer's V <https://en.wikipedia.org/wiki/Cramér%27s_V>`_

    Pearson's Correlation Coefficient is a measure of the linear correlation
    between two variables, giving a value between -1 and +1 (inclusive).

    * `Pearson's Correlation Coefficient
      <https://en.wikipedia.org/wiki/Pearson_correlation_coefficient>`_

    * `pandas.DataFrame.corr
      <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html>`_

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import skrub
    >>> pd.set_option('display.width', 200)
    >>> pd.set_option('display.max_columns', 10)
    >>> pd.set_option('display.precision', 4)
    >>> rng = np.random.default_rng(33)
    >>> df = pd.DataFrame({f"c_{i}": rng.random(size=20)*10 for i in range(5)})
    >>> df["c_str"] = [f"val {i}" for i in range(df.shape[0])]
    >>> df.shape
    (20, 6)
    >>> df.head()
          c_0     c_1     c_2     c_3     c_4  c_str
    0  4.4364  4.0114  6.9271  7.0970  4.8913  val 0
    1  5.6849  0.7192  7.6430  4.6441  2.5116  val 1
    2  9.0810  9.4011  1.9257  5.7429  6.2358  val 2
    3  2.5425  2.9678  9.7801  9.9879  6.0709  val 3
    4  5.8878  9.3223  5.3840  7.2006  2.1494  val 4
    >>> # Compute the associations
    >>> associations = skrub.column_associations(df)
    >>> associations # doctest: +SKIP
       left_column_name  left_column_idx right_column_name  right_column_idx  cramer_v  pearson_corr
    0              c_1                1               c_4                 4    0.8215        0.1597
    1              c_0                0               c_1                 1    0.8215        0.1123
    2              c_0                0               c_3                 3    0.7551        0.3212
    3              c_1                1               c_3                 3    0.6837       -0.1887
    4              c_0                0               c_4                 4    0.6837       -0.3202
    5              c_3                3               c_4                 4    0.6053       -0.0150
    6              c_2                2               c_3                 3    0.6053        0.1757
    7              c_0                0               c_2                 2    0.6053       -0.0578
    8              c_2                2               c_4                 4    0.5169       -0.2885
    9              c_1                1               c_2                 2    0.4122       -0.4986
    >>> pd.reset_option('display.width')
    >>> pd.reset_option('display.max_columns')
    >>> pd.reset_option('display.precision')
    """  # noqa: E501
    cramer_v_table = _stack_symmetric_associations(
        _cramer_v_matrix(df),
        df,
    )
    pearson_c_table = _compute_pearson(df)
    on = ["left_column_name", "right_column_name"]
    return _join_utils.left_join(
        cramer_v_table, pearson_c_table, right_on=on, left_on=on
    )


def _stack_symmetric_associations(associations, df):
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
    col_names = np.asarray(list(map(str, sbd.column_names(df))))
    left_column_names, right_column_names = (
        col_names[left_indices],
        col_names[right_indices],
    )
    result = {
        "left_column_name": left_column_names,
        "left_column_idx": left_indices,
        "right_column_name": right_column_names,
        "right_column_idx": right_indices,
        "cramer_v": associations,
    }
    return sbd.make_dataframe_like(df, result)


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
    for col_idx in range(n_cols):
        col = sbd.col_by_idx(df, col_idx)
        if sbd.is_duration(col):
            col = sbd.total_seconds(col)
        if sbd.is_numeric(col) or sbd.is_any_date(col):
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
    if np.issubdtype(values.dtype, np.floating):
        # OneHotEncoder allows NaN but not inf , so clip to finite values when
        # the input is floating-point
        finfo = np.finfo(values.dtype)
        values = np.clip(values, finfo.min, finfo.max)
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
    """Compute the Cramer's V statistic given a contingency table.

    The input is the table computed by ``_contingency_table`` with shape
    (n cols, n cols, n bins, n bins).

    This returns the symmetric matrix with shape (n cols, n cols) where entry
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


def _compute_pearson(df):
    """Compute the Pearson's correlation coefficient for all pairs of columns.

    This returns the Pearson's correlation as a dataframe in the long format, of shape
    (n samples * n samples, 3), whose module is the same as the input.
    """
    corr = sbd.pearson_corr(df)
    if sbd.shape(corr)[0] == 0:
        return sbd.make_dataframe_like(
            df,
            {
                "left_column_name": np.array([], dtype="str"),
                "right_column_name": np.array([], dtype="str"),
                "pearson_corr": np.array([], dtype="float64"),
            },
        )
    return _melt(
        corr,
        left_col="left_column_name",
        right_col="right_column_name",
        val="pearson_corr",
    )


@sbd._common.dispatch
def _melt(df, left_col, right_col, val):
    raise NotImplementedError()


@_melt.specialize("pandas", argument_type="DataFrame")
def _melt_pandas(df, left_col, right_col, val):
    # Deal with multi-level index and columns
    if df.index.nlevels > 1:
        df.index = df.index.to_flat_index().astype("string")
    if df.columns.nlevels > 1:
        df.columns = df.columns.to_flat_index().astype("string")
    return df.melt(ignore_index=False, var_name=right_col, value_name=val).reset_index(
        names=left_col
    )


@_melt.specialize("polars", argument_type="DataFrame")
def _melt_polars(df, left_col, right_col, val):
    df = df.transpose(
        include_header=True, header_name=left_col, column_names=df.columns
    )
    # TODO: remove when polars optional min-dep > 1.0
    import polars as pl

    if parse_version(pl.__version__) < parse_version("1.0"):
        return df.melt(id_vars=left_col, variable_name=right_col, value_name=val)
    else:
        return df.unpivot(index=left_col, variable_name=right_col, value_name=val)
