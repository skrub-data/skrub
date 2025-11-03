"""Detect which columns have strong statistical dependence using InterDependenceScore"""

import math
import sys
from typing import Tuple, TypeVar, Union

import numpy
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from . import _dataframe as sbd

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
        x : numpy.ndarray
            An array of shape (n_samples, n_features).

        k_terms : int
            The number of terms to compute.

        bandwidth_term : float
            The gaussian bandwidth determines the smoothness of the obtained functions

    Returns
    -------
        result : numpy.ndarray
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
        X: numpy.ndarray
            An array of shape (n_samples, n_features).

    Returns
    -------
        result: numpy.ndarray
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
        output: numpy.ndarray
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
    norm="inf",
    n_tests=100,
    bandwidth_term=1 / 2,
) -> numpy.ndarray:
    """
    Compute the p-value for the interdependence score between two variables.

    Parameters
    ----------
        ids: numpy.ndarray
            An array of shape (n_features, n_features).
            The interdependence score matrix computed.

        X: numpy.ndarray
            An array of shape (n_samples, n_features).

        feature_map_function: function
            The feature map function

        k_terms: int
            The number of terms to compute for the feature map function

        norm: str
            The order of the vector norm (1, 2, inf)

        n_tests: int
            The number of tests to run in order to compute the p-value.

        bandwidth_term: float
            The gaussian bandwidth

    Returns
    -------
        p_vals: numpy.ndarray
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
            norm=norm,
            bandwidth_term=bandwidth_term,
        )
        count += np.where(ids_perm > ids_obs, 1, 0)
    p_vals = count / n_tests
    return p_vals


def _apply_norm(C: numpy.ndarray, norm: str) -> numpy.ndarray:
    """
    Compute the norm for IDS score

    Parameters
    ----------
        C: numpy.ndarray
            An array of shape (k_terms, n_features, k_terms, n_features).
            It is the correlation between the feature map component of each feature.

        norm: str
            The order of the vector norm (1, 2, inf)

    Returns
    -------
        res: numpy.ndarray
            An array of shape (n_features, n_features).
            The interdependence score matrix.
    """
    if norm == "inf":
        return np.max(np.abs(C), axis=(0, 2))
    elif norm == 2:
        return np.sqrt(np.mean(np.abs(C) ** 2, axis=(0, 1)))
    elif norm == 1:
        return np.mean(np.abs(C), axis=(0, 1))
    else:
        raise ValueError(f"Unsupported norm type: {norm}")


def _compute_ids_numpy(
    X: numpy.ndarray,
    feature_map_function=_gaussian_feature_map,
    k_terms=6,
    norm="inf",
    p_val=False,
    num_tests=100,
    bandwidth_term=1 / 2,
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, None]]:
    """
    Compute the InterDependence score matrix between the columns of X.

    Parameters
    ----------
        X: numpy.ndarray
            An array of shape (n_samples, n_features).

        feature_map_function: function
            The feature map function

        k_terms: int
            The number of terms to compute for the feature map function

        norm: str
            The order of the vector norm (1, 2, inf)

        p_val: bool
            True if the p-values should be computed.

        num_tests: int
            The number of tests to run in order to compute the p-value.

        bandwidth_term: float
            The gaussian bandwidth

    Returns
    -------
        ids: numpy.ndarray
            An array of shape (n_features, n_features).
            The interdependence score matrix.
        pvals: numpy.ndarray
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
    ids = _apply_norm(C, norm)

    if p_val:
        p_vals = _compute_p_val(
            ids, X, feature_map_function, k_terms, norm, num_tests, bandwidth_term
        )
        return ids, p_vals
    return ids, None


def merge_one_hot_features(
    arr: numpy.ndarray, features_match: dict[int, list], how="mean"
):
    """
    Merge the one hot features in order to get IDS matrix for the initial features

    Parameters
    ----------
        arr: numpy.ndarray
            An array of shape (n_features, n_features).
            The interdependence score matrix.
        features_match: dict
            Dictionary mapping the index of the initial features to the indices
            of their one-hot components or to their new position in the one hot array.
        how: str
            The way to aggregate the values of the one hot features
            to get the estimated ids score of the original feature.
            One of 'mean', 'max'

    Returns
    -------
        arr: numpy.ndarray
            An array of shape (n_initial_features, n_initial_features).
    """

    original_number_feats = len(features_match.keys())

    for original_feature_idx in features_match.keys():
        indices_to_merge = features_match[original_feature_idx]
        if len(indices_to_merge) == 1:
            arr[original_feature_idx, :] = arr[indices_to_merge[0], :]
        else:
            if how == "max":
                arr[original_feature_idx, :] = arr[indices_to_merge, :].max(axis=0)
            else:
                arr[original_feature_idx, :] = arr[indices_to_merge, :].mean(axis=0)
    arr = arr[:original_number_feats, :]

    for original_feature_idx in features_match.keys():
        indices_to_merge = features_match[original_feature_idx]
        if len(indices_to_merge) == 1:
            arr[:, original_feature_idx] = arr[:, indices_to_merge[0]]
        else:
            if how == "max":
                arr[:, original_feature_idx] = arr[:, indices_to_merge].max(axis=1)
            else:
                arr[:, original_feature_idx] = arr[:, indices_to_merge].mean(axis=1)
    arr = arr[:, :original_number_feats]
    return arr


def _ids_matrix(
    X: DataFrame,
    feature_map_function=_gaussian_feature_map,
    k_terms=6,
    norm="inf",
    p_val=False,
    num_tests=100,
    bandwidth_term=1 / 2,
    squeeze_option="mean",
) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, None]]:
    """
        Compute the interdependence score matrix.

    Parameters
    ----------
        X: DataFrame
            An array of shape (n_samples, n_features).

        feature_map_function: function
            The feature map function

        k_terms: int
            The number of terms to compute for the feature map function

        norm: str
            The order of the vector norm (1, 2, inf)

        p_val: bool
            True if the p-values should be computed.

        num_tests: int
            The number of tests to run in order to compute the p-value.

        bandwidth_term: float
            The gaussian bandwidth

        squeeze_option: str
            How to aggregate the one-hot encoded features to get the initial features.

    Returns
    -------
        ids_matrix: numpy.ndarray
            The interdependence score matrix.
        pval_matrix: numpy.ndarray
            The p-value matrix.
    """
    X_encoded, features_match = _onehot_encode_categoricals(X)
    ids_matrix, pval_matrix = _compute_ids_numpy(
        X_encoded,
        feature_map_function=feature_map_function,
        k_terms=k_terms,
        norm=norm,
        p_val=p_val,
        num_tests=num_tests,
        bandwidth_term=bandwidth_term,
    )

    ids_matrix = merge_one_hot_features(ids_matrix, features_match, how=squeeze_option)
    if p_val:
        pval_matrix = merge_one_hot_features(
            pval_matrix, features_match, how=squeeze_option
        )
    if p_val:
        return ids_matrix, pval_matrix
    return ids_matrix, None
