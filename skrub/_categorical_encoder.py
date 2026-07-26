"""Hybrid categorical encoder for frequent and rare categories.

This module implements a single-column categorical encoder that combines:

- One-hot encoding for frequent categories.
- Target encoding in one shared numerical column for rare categories.

The encoder supports both pandas and polars columns through skrub's dispatch
mechanism.

Notes
-----
``fit_transform`` uses leave-one-out target encoding for rare categories on the
training data. This avoids encoding each training observation with a target mean
that directly contains its own target value. ``transform`` uses the category
means learned on the full training set and falls back to the global target mean
for unseen rare categories.
"""

from __future__ import annotations

import numbers

import numpy as np
import pandas as pd

from skrub._dispatch import dispatch, raise_dispatch_unregistered_type


@dispatch
def _get_col_name(col):
    raise_dispatch_unregistered_type(col, kind="Series")


@_get_col_name.specialize("pandas", argument_type="Column")
def _get_col_name_pandas(col):
    return col.name if col.name is not None else "col"


@_get_col_name.specialize("polars", argument_type="Column")
def _get_col_name_polars(col):
    return col.name if col.name is not None else "col"


@dispatch
def _value_counts(col):
    raise_dispatch_unregistered_type(col, kind="Series")


@_value_counts.specialize("pandas", argument_type="Column")
def _value_counts_pandas(col):
    return col.value_counts(dropna=False).to_dict()


@_value_counts.specialize("polars", argument_type="Column")
def _value_counts_polars(col):
    result = col.value_counts()
    return dict(zip(result[col.name].to_list(), result["count"].to_list()))


@dispatch
def _to_numpy_1d(col):
    raise_dispatch_unregistered_type(col, kind="Series")


@_to_numpy_1d.specialize("pandas", argument_type="Column")
def _to_numpy_1d_pandas(col):
    return col.to_numpy()


@_to_numpy_1d.specialize("polars", argument_type="Column")
def _to_numpy_1d_polars(col):
    return col.to_numpy()


@dispatch
def _make_ohe_col(col, category):
    raise_dispatch_unregistered_type(col, kind="Series")


@_make_ohe_col.specialize("pandas", argument_type="Column")
def _make_ohe_col_pandas(col, category):
    return (col == category).astype(int).to_numpy()


@_make_ohe_col.specialize("polars", argument_type="Column")
def _make_ohe_col_polars(col, category):
    return (col == category).cast(int).to_numpy()


@dispatch
def _make_dataframe(col, data):
    """Return an output DataFrame using the same backend as ``col``."""
    raise_dispatch_unregistered_type(col, kind="Series")


@_make_dataframe.specialize("pandas", argument_type="Column")
def _make_dataframe_pandas(col, data):
    return pd.DataFrame(data, index=col.index)


@_make_dataframe.specialize("polars", argument_type="Column")
def _make_dataframe_polars(col, data):
    import polars as pl

    return pl.DataFrame(data)


def _is_missing(value):
    """Return whether a scalar category value is missing."""
    try:
        result = pd.isna(value)
        return bool(result) if np.ndim(result) == 0 else False
    except (TypeError, ValueError):
        return False


def _same_category(left, right):
    """Compare category values while treating missing values as equal."""
    if _is_missing(left) and _is_missing(right):
        return True
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False


def _category_mask(values, category):
    """Boolean mask selecting ``category`` from a 1D object array."""
    if _is_missing(category):
        return np.asarray([_is_missing(value) for value in values], dtype=bool)
    return np.asarray([_same_category(value, category) for value in values], dtype=bool)


def _safe_category_label(category):
    """Convert a category value to a stable feature-name component."""
    return "missing" if _is_missing(category) else str(category)


class CategoricalEncoder:
    """Hybrid one-hot / target encoder for one categorical column.

    Categories observed at least ``min_frequency`` times are represented by
    individual one-hot columns. All remaining categories share a single
    target-encoded column.

    Parameters
    ----------
    min_frequency : int, default=30
        Minimum number of training observations required for a category to be
        considered frequent and therefore one-hot encoded.

    Attributes
    ----------
    frequent_categories_ : list
        Categories represented by one-hot columns.

    rare_categories_ : list
        Categories represented through the shared target-encoding column.

    target_encoder_ : dict
        Mapping from rare categories to their target means. The key
        ``"__default__"`` stores the global target mean used for unseen
        categories.

    all_outputs_ : list of str
        Output feature names.

    Notes
    -----
    ``fit_transform`` uses leave-one-out target means for rare categories on
    the training observations to reduce target leakage. ``transform`` uses
    target means estimated on the full training sample.
    """

    def __init__(self, min_frequency=30):
        self.min_frequency = min_frequency

    def _validate_params(self):
        if (
            not isinstance(self.min_frequency, numbers.Integral)
            or isinstance(self.min_frequency, bool)
            or self.min_frequency < 1
        ):
            raise ValueError(
                "min_frequency must be an integer greater than or equal to 1."
            )

    def _check_is_fitted(self):
        required_attributes = (
            "frequent_categories_",
            "rare_categories_",
            "target_encoder_",
            "global_target_mean_",
            "_col_name",
            "all_outputs_",
        )
        if not all(hasattr(self, attr) for attr in required_attributes):
            raise ValueError(
                "CategoricalEncoder is not fitted yet. "
                "Call fit() or fit_transform() before transform()."
            )

    def fit(self, col, y):
        """Learn frequent categories and rare-category target means."""
        self._validate_params()

        col_values = np.asarray(_to_numpy_1d(col), dtype=object)
        y_values = np.asarray(_to_numpy_1d(y), dtype=float)

        if col_values.ndim != 1 or y_values.ndim != 1:
            raise ValueError("col and y must both be one-dimensional.")
        if len(col_values) != len(y_values):
            raise ValueError("col and y must contain the same number of observations.")
        if len(col_values) == 0:
            raise ValueError("Cannot fit CategoricalEncoder on an empty column.")
        if np.isnan(y_values).any():
            raise ValueError("y must not contain missing values.")

        value_counts = _value_counts(col)
        categories = list(value_counts.keys())

        frequent_categories = [
            category
            for category in categories
            if value_counts[category] >= self.min_frequency
        ]
        rare_categories = [
            category
            for category in categories
            if value_counts[category] < self.min_frequency
        ]

        try:
            frequent_categories = sorted(frequent_categories)
        except TypeError:
            frequent_categories = sorted(frequent_categories, key=str)

        try:
            rare_categories = sorted(rare_categories)
        except TypeError:
            rare_categories = sorted(rare_categories, key=str)

        global_mean = float(np.mean(y_values))

        target_encoder = {"__default__": global_mean}
        category_target_sum = {}
        category_target_count = {}

        for category in rare_categories:
            mask = _category_mask(col_values, category)
            count = int(mask.sum())
            target_sum = float(y_values[mask].sum())

            category_target_sum[category] = target_sum
            category_target_count[category] = count
            target_encoder[category] = target_sum / count if count > 0 else global_mean

        self.frequent_categories_ = frequent_categories
        self.rare_categories_ = rare_categories
        self.onehot_encoder_ = {
            category: i for i, category in enumerate(frequent_categories)
        }
        self.target_encoder_ = target_encoder
        self.global_target_mean_ = global_mean
        self._category_target_sum_ = category_target_sum
        self._category_target_count_ = category_target_count
        self._col_name = _get_col_name(col)

        self.all_outputs_ = [
            f"{self._col_name}__{_safe_category_label(category)}"
            for category in frequent_categories
        ] + [f"{self._col_name}__rare_target"]

        return self

    def transform(self, col):
        """Transform a new categorical column using the fitted encoder."""
        self._check_is_fitted()

        col_values = np.asarray(_to_numpy_1d(col), dtype=object)
        data = {}

        for category in self.frequent_categories_:
            feature_name = f"{self._col_name}__{_safe_category_label(category)}"
            data[feature_name] = _make_ohe_col(col, category)

        rare_target = []

        for value in col_values:
            if any(
                _same_category(value, category)
                for category in self.frequent_categories_
            ):
                rare_target.append(0.0)
                continue

            encoded = self.global_target_mean_
            for category, target_mean in self.target_encoder_.items():
                if category == "__default__":
                    continue
                if _same_category(value, category):
                    encoded = target_mean
                    break

            rare_target.append(float(encoded))

        data[f"{self._col_name}__rare_target"] = np.asarray(rare_target, dtype=float)

        return _make_dataframe(col, data)

    def fit_transform(self, col, y):
        """Fit and transform the training column with leakage reduction.

        Rare training categories use leave-one-out target encoding whenever
        possible. Singleton categories fall back to the global target mean.
        """
        self.fit(col, y)

        col_values = np.asarray(_to_numpy_1d(col), dtype=object)
        y_values = np.asarray(_to_numpy_1d(y), dtype=float)
        data = {}

        for category in self.frequent_categories_:
            feature_name = f"{self._col_name}__{_safe_category_label(category)}"
            data[feature_name] = _make_ohe_col(col, category)

        rare_target = np.zeros(len(col_values), dtype=float)

        for row_idx, (value, target_value) in enumerate(zip(col_values, y_values)):
            if any(
                _same_category(value, category)
                for category in self.frequent_categories_
            ):
                rare_target[row_idx] = 0.0
                continue

            matched_category = None
            for category in self.rare_categories_:
                if _same_category(value, category):
                    matched_category = category
                    break

            if matched_category is None:
                rare_target[row_idx] = self.global_target_mean_
                continue

            count = self._category_target_count_[matched_category]
            target_sum = self._category_target_sum_[matched_category]

            if count > 1:
                rare_target[row_idx] = (target_sum - float(target_value)) / (count - 1)
            else:
                rare_target[row_idx] = self.global_target_mean_

        data[f"{self._col_name}__rare_target"] = rare_target

        return _make_dataframe(col, data)

    def get_feature_names_out(self):
        """Return output feature names learned during fitting."""
        self._check_is_fitted()
        return np.asarray(self.all_outputs_, dtype=object)
