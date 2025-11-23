"""Detect which columns have strong statistical dependence using InterDependenceScore"""

import math
import sys
from typing import Tuple, TypeVar, Union

import numpy
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from . import _dataframe as sbd
from . import _join_utils
from ._column_associations import _stack_symmetric_associations

DataFrame = TypeVar("DataFrame")
_N_BINS = 10
_CATEGORICAL_THRESHOLD = 30
EPSILON = sys.float_info.epsilon


def _gaussian_feature_map(
    x: numpy.ndarray, k_terms=6, bandwidth_term=1 / 2
) -> numpy.ndarray:
    """
    Compute the feature map associated with the approximated gaussian kernel.
    For each element of the matrix x, the k terms of the feature map are computed.
    Parameters
    ----------
        x : numpy array
            An array of shape (n_samples, n_features).

        k_terms : int
            The number of terms to compute.

        bandwidth_term : float
            The gaussian bandwidth determines the smoothness of the obtained functions

    Returns
    -------
        result : numpy array
            The resulting array of shape (n_samples, n_features * k_terms).
    """
    B = bandwidth_term
    exp = np.exp(-B * x**2)
    terms = []
    for i in range(k_terms):
        # In their implementation the authors of IDS chose to set B = 1.
        terms.append(exp * x**i / (math.sqrt(math.factorial(i)) * 1.0))
    return np.concatenate(terms, axis=1)


def _center(X: numpy.ndarray) -> numpy.ndarray:
    return X - np.mean(X, axis=0, keepdims=True)


def _permute_rows(X: numpy.ndarray) -> numpy.ndarray:
    """
    Permute the values of the rows for each column of X.
    It is a random permutation.

    Parameters
    ----------
        X: numpy array
            An array of shape (n_samples, n_features).

    Returns
    -------
        result: numpy array
    """
    n, d = X.shape
    rng = np.random.default_rng()
    result = np.empty_like(X)
    for j in range(d):
        result[:, j] = X[rng.permutation(n), j]
    return result


def _onehot_encode_categoricals(df: DataFrame) -> tuple[DataFrame, dict[int, list]]:
    """
    Encode the categorical variables into one-hot encoding.
    Numerical variables which values are fewer than _CATEGORICAL_THRESHOLD are treated
    as categorical.

    Parameters
    ----------
        df: DataFrame

    Returns
    -------
        output: numpy array
            The resulting array with one hot encoded features.
        old_idx_matches: dict[int, list]
    """
    _, n_cols = sbd.shape(df)
    new_cols = []
    old_idx_matches = dict()
    one_hot_encoder = OneHotEncoder(max_categories=_N_BINS, sparse_output=False)
    current_col_idx = 0
    for col_idx in range(n_cols):
        col = sbd.col_by_idx(df, col_idx)
        one_hot_cols = None
        if sbd.is_duration(col):
            col = sbd.total_seconds(col)
        if sbd.is_numeric(col) or sbd.is_any_date(col):
            col = sbd.to_float32(col)
            if sbd.n_unique(col) >= _CATEGORICAL_THRESHOLD:
                new_cols.append(col.to_numpy().reshape(-1, 1))
            else:
                # OneHotEncoder requires numeric values to be sorted.
                col = np.sort(col.to_numpy()).reshape(-1, 1)
                if col.dtype != np.object_:
                    finfo = np.finfo(col.dtype)
                    col = np.clip(col, finfo.min, finfo.max)
                one_hot_cols = one_hot_encoder.fit_transform(col)
        else:
            col = sbd.to_string(col).to_numpy().reshape(-1, 1)
            one_hot_cols = one_hot_encoder.fit_transform(col)

        if one_hot_cols is not None:
            new_cols.append(one_hot_cols)
            old_idx_matches[col_idx] = [
                i
                for i in range(current_col_idx, current_col_idx + one_hot_cols.shape[1])
            ]
            current_col_idx += one_hot_cols.shape[1]
        else:
            old_idx_matches[col_idx] = [current_col_idx]
            current_col_idx += 1

    output = np.hstack(new_cols)
    return output, old_idx_matches


def _compute_p_val(
    ids: numpy.ndarray,
    X: numpy.ndarray,
    feature_map_function=_gaussian_feature_map,
    k_terms=6,
    n_tests=100,
    bandwidth_term=1 / 2,
) -> numpy.ndarray:
    """
    Compute the p-value for the interdependence score between two variables.

    Parameters
    ----------
        ids: numpy array
            An array of shape (n_features, n_features).
            The interdependence score matrix computed.

        X: numpy array
            An array of shape (n_samples, n_features).

        feature_map_function: function
            The feature map function

        k_terms: int
            The number of terms to compute for the feature map function

        n_tests: int
            The number of tests to run in order to compute the p-value.

        bandwidth_term: float
            The gaussian bandwidth

    Returns
    -------
        p_vals: numpy array
            The p-value matrix associated to the interdependence score matrix.
    """
    ids_obs = ids
    count = np.zeros((X.shape[1], X.shape[1]))
    for i in range(n_tests):
        X_permuted = _permute_rows(X)
        ids_perm, _ = _compute_ids_numpy(
            X_permuted,
            k_terms=k_terms,
            feature_map_function=feature_map_function,
            bandwidth_term=bandwidth_term,
        )
        count += np.where(ids_perm > ids_obs, 1, 0)
    p_vals = count / n_tests
    return p_vals


def _compute_ids_numpy(
    X: numpy.ndarray,
    feature_map_function=_gaussian_feature_map,
    k_terms=6,
    p_val=False,
    num_tests=100,
    bandwidth_term=1 / 2,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, None]]:
    """
    Compute the InterDependence score matrix between the columns of X.

    Parameters
    ----------
        X: numpy array
            An array of shape (n_samples, n_features).

        feature_map_function: function
            The feature map function

        k_terms: int
            The number of terms to compute for the feature map function

        p_val: bool
            True if the p-values should be computed.

        num_tests: int
            The number of tests to run in order to compute the p-value.

        bandwidth_term: float
            The gaussian bandwidth

    Returns
    -------
        ids: numpy array
            An array of shape (n_features, n_features).
            The interdependence score matrix.

        pvals: numpy array
            An array of shape (n_features, n_features).
            The p-value matrix associated to the interdependence score matrix.
    """
    dx = X.shape[1]
    X_transformed = feature_map_function(
        X, k_terms=k_terms, bandwidth_term=bandwidth_term
    )
    X_transformed = _center(X_transformed)

    C = np.corrcoef(X_transformed, rowvar=False)
    C = C.reshape(k_terms, dx, k_terms, dx)

    C = np.nan_to_num(C, nan=0, posinf=0, neginf=0)
    # Apply infinity-norm
    ids = np.max(np.abs(C), axis=(0, 2))

    if p_val:
        p_vals = _compute_p_val(
            ids, X, feature_map_function, k_terms, num_tests, bandwidth_term
        )
    else:
        p_vals = None
    return ids, p_vals


def merge_one_hot_features(arr: numpy.ndarray, features_match: dict[int, list]):
    """
    Merge the one hot features in order to get IDS matrix for the initial features

    Parameters
    ----------
        arr: numpy array
            An array of shape (n_features, n_features).
            The interdependence score matrix.
        features_match: dict
            Dictionary mapping the index of the initial features to the indices
            of their one-hot components or to their new position in the one hot array.

    Returns
    -------
        arr: numpy array
            An array of shape (n_initial_features, n_initial_features).
    """

    original_number_feats = len(features_match.keys())
    for original_feature_idx in features_match.keys():
        indices_to_merge = features_match[original_feature_idx]
        if len(indices_to_merge) == 1:
            arr[original_feature_idx, :] = arr[indices_to_merge[0], :]
        else:
            arr[original_feature_idx, :] = arr[indices_to_merge, :].max(axis=0)
    arr = arr[:original_number_feats, :]

    for original_feature_idx in features_match.keys():
        indices_to_merge = features_match[original_feature_idx]
        if len(indices_to_merge) == 1:
            arr[:, original_feature_idx] = arr[:, indices_to_merge[0]]
        else:
            arr[:, original_feature_idx] = arr[:, indices_to_merge].max(axis=1)
    arr = arr[:, :original_number_feats]
    return arr


def _ids_matrix(
    X: DataFrame,
    feature_map_function=_gaussian_feature_map,
    k_terms=6,
    p_val=False,
    num_tests=100,
    bandwidth_term=1 / 2,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, None]]:
    """
        Compute the interdependence score matrix.

    Parameters
    ----------
        X: DataFrame
            An dataframe of shape (n_samples, n_features).

        feature_map_function: function
            The feature map function

        k_terms: int
            The number of terms to compute for the feature map function

        p_val: bool
            True if the p-values should be computed.

        num_tests: int
            The number of tests to run in order to compute the p-value.

        bandwidth_term: float
            The gaussian bandwidth

    Returns
    -------
        ids_matrix: numpy array,
            The interdependence score matrix (n_features, n_features).

        pval_matrix: numpy array
            The p-value matrix (n_features, n_features).
    """
    X_encoded, features_match = _onehot_encode_categoricals(X)
    ids_matrix, pval_matrix = _compute_ids_numpy(
        X_encoded,
        feature_map_function=feature_map_function,
        k_terms=k_terms,
        p_val=p_val,
        num_tests=num_tests,
        bandwidth_term=bandwidth_term,
    )

    ids_matrix = merge_one_hot_features(ids_matrix, features_match)
    if p_val:
        pval_matrix = merge_one_hot_features(pval_matrix, features_match)
    else:
        pval_matrix = None

    return ids_matrix, pval_matrix


def interdependence_score(
    X: DataFrame,
    k_terms=6,
    p_val=False,
    num_tests=100,
    bandwidth_term=1 / 2,
    return_matrix=False,
) -> Union[DataFrame, Tuple[DataFrame, DataFrame]]:
    r"""
    Compute the InterDependence Score (IDS).

    The InterDependence Score is a measure of dependence that captures linear and
    various nonlinear dependencies between variables. The IDS algorithm is based on
    a dependence measure defined in infinite-dimensional Hilbert spaces, capable of capturing
    any type of dependence, and a fast algorithm that neural networks natively implement
    to compute dependencies between random variables.
    IDS range from 0 to 1 (high dependence).

    Parameters
    ----------
    X : DataFrame
        An dataframe of shape (n_samples, n_features).

    k_terms : int, default=6
        The number of terms `k` to compute for the feature map associated to universal Gaussian kernel.
        A canonical feature map for this kernel is :
        .. math::
            \begin{align*}
                \phi(x) = exp\left(-x^2/(2B^2)\right) \left[1, x/B, \cdots, x^k/\left(\sqrt{k!}\, B^k\right), \cdots\right]
            \end{align*}

    p_val : bool, default=False
        If True, the p-value associated to each score is compute and added to the output table.

    num_tests : int, default=100
        The number of tests to run in order to compute the p-values.

    bandwidth_term : float, default=0.5
        The gaussian bandwidth that represents the term `B` in the feature map formula.

    return_matrix : bool, default=False
        If True, return the squared matrix of interdependence scores (n_features, n_features).

    Returns
    -------
    DataFrame or (DataFrame, DataFrame)
        If ``return_matrix=False`` (default), returns a DataFrame containing each
        pair of columns, their interdependence score, and optionally the p-value.
        If ``return_matrix=True``, returns also a square matrix of interdependence scores.

    Notes
    -----
    The result is a dataframe with columns: ``['left_column_name', 'right_column_name', 'cramer_v', 'pearson_corr']``

    Each pair of columns appears only once, and the scores are sorted in descending order.

    Numeric columns with fewer than 30 values are treated as categorical.
    The categoricals columns are one-hot encoded.

    The implemented IDS only use the infinity norm.

    References
    ----------
    This measure of dependence has been introduced in the paper `Efficiently quantifying dependence
    in massive scientific datasets using InterDependence Scores
    (A. Radhakrishnan, Y. Jain, C. Uhler, & E.S. Lander)<https://doi.org/10.1073/pnas.2509860122>`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from skrub import interdependence_score
    >>> rng = np.random.default_rng(42)
    >>> x = rng.random(1000)*13
    >>> x_log = np.log(x)
    >>> x_square = np.square(x)
    >>> z = rng.random(1000)
    >>> features = np.column_stack([x, x_log, x_square, z])
    >>> df = pd.DataFrame(features, columns=['x', 'log(x)', 'x²', 'z'])
    >>> df.head()
               x    log(x)          x²         z
    0  10.061429  2.308709  101.232346  0.062063
    1   5.705420  1.741417   32.551814  0.458262
    2  11.161773  2.412495  124.585176  0.129030
    3   9.065784  2.204507   82.188446  0.152327
    4   1.224306  0.202374    1.498924  0.632283
    >>> table = interdependence_score(df, p_val=True, num_tests=100)
    >>> table
      left_column_name right_column_name  interdependence_score  pvalue
    0                x                x²               0.966105    0.00
    1                x            log(x)               0.889642    0.00
    2           log(x)                x²               0.776569    0.00
    3           log(x)                 z               0.042476    0.76
    4               x²                 z               0.042180    0.67
    5                x                 z               0.036878    0.65
    """  # noqa: E501

    ids_matrix, pval_matrix = _ids_matrix(
        X,
        feature_map_function=_gaussian_feature_map,
        k_terms=k_terms,
        p_val=p_val,
        num_tests=num_tests,
        bandwidth_term=bandwidth_term,
    )

    ids_table = _stack_symmetric_associations(
        ids_matrix, X, statistic_name="interdependence_score"
    )

    if p_val:
        pval_table = _stack_symmetric_associations(
            pval_matrix, X, statistic_name="pvalue"
        )
        on = ["left_column_name", "right_column_name"]
        ids_table = _join_utils.left_join(
            ids_table, pval_table, right_on=on, left_on=on
        )
        ids_table = drop_columns_with_substring(ids_table, "skrub")

    ids_table = drop_columns_with_substring(ids_table, "idx")
    if return_matrix:
        ids_matrix_df = sbd.make_dataframe_like(
            X,
            {
                col_name: ids_matrix[:, i]
                for i, col_name in enumerate(sbd.column_names(X))
            },
        )
        return ids_table, ids_matrix_df
    else:
        return ids_table


@sbd._common.dispatch
def drop_columns_with_substring(df, substring):
    """Drop columns whose name contains the specified substring."""
    raise NotImplementedError()


@drop_columns_with_substring.specialize("pandas")
def _drop_columns_containing_pandas(df, substring):
    cols_to_keep = [col for col in df.columns if substring not in col]
    return df[cols_to_keep]


@drop_columns_with_substring.specialize("polars")
def _drop_columns_containing_polars(df, substring):
    cols_to_keep = [col for col in df.columns if substring not in col]
    return df.select(cols_to_keep)
