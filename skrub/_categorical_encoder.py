from skrub._dispatch import dispatch
from skrub._dispatch import raise_dispatch_unregistered_type

import pandas as pd
import numpy as np


# =========================
# Dispatch helpers
# =========================

@dispatch
def _get_col_name(col):
    raise_dispatch_unregistered_type(col, kind="Series")


@_get_col_name.specialize("pandas", argument_type="Column")
def _get_col_name_pandas(col):
    return col.name if col.name is not None else "col"


@_get_col_name.specialize("polars", argument_type="Column")
def _get_col_name_polars(col):
    return col.name


@dispatch
def _value_counts(col):
    raise_dispatch_unregistered_type(col, kind="Series")


@_value_counts.specialize("pandas", argument_type="Column")
def _value_counts_pandas(col):
    return col.value_counts().to_dict()


@_value_counts.specialize("polars", argument_type="Column")
def _value_counts_polars(col):
    result = col.value_counts()
    return dict(zip(result[col.name].to_list(), result["count"].to_list()))


@dispatch
def _make_ohe_col(col, cat):
    raise_dispatch_unregistered_type(col, kind="Series")


@_make_ohe_col.specialize("pandas", argument_type="Column")
def _make_ohe_col_pandas(col, cat):
    return (col == cat).astype(int).to_numpy()


@_make_ohe_col.specialize("polars", argument_type="Column")
def _make_ohe_col_polars(col, cat):
    return (col == cat).cast(int).to_numpy()


@dispatch
def _make_target_col(col, target_encoder, frequent_set, default_val):
    raise_dispatch_unregistered_type(col, kind="Series")


@_make_target_col.specialize("pandas", argument_type="Column")
def _make_target_col_pandas(col, target_encoder, frequent_set, default_val):
    return [
        target_encoder.get(v, default_val) if v not in frequent_set else 0.0
        for v in col
    ]


@_make_target_col.specialize("polars", argument_type="Column")
def _make_target_col_polars(col, target_encoder, frequent_set, default_val):
    return [
        target_encoder.get(v, default_val) if v not in frequent_set else 0.0
        for v in col.to_list()
    ]


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
def _make_dataframe(col, data_dict):
    raise_dispatch_unregistered_type(col, kind="Series")


@_make_dataframe.specialize("pandas", argument_type="Column")
def _make_dataframe_pandas(col, data_dict):
    result = pd.DataFrame(data_dict)
    result.index = col.index
    return result


@_make_dataframe.specialize("polars", argument_type="Column")
def _make_dataframe_polars(col, data_dict):
    import polars as pl
    return pl.DataFrame(data_dict)


# =========================
# CategoricalEncoder
# =========================

class CategoricalEncoder:
    """
    Encode une colonne catégorielle :

    - Catégories fréquentes >= max_frequent occurrences :
      One-Hot Encoding

    - Catégories rares < max_frequent occurrences :
      Target Encoding dans une seule colonne

    Supporte pandas et polars via @dispatch.
    """

    def __init__(self, max_frequent=30):
        self.max_frequent = max_frequent

    def fit(self, col, y):
        col_name = _get_col_name(col)
        value_counts = _value_counts(col)

        y_numpy = _to_numpy_1d(y).astype(float)
        global_mean = float(y_numpy.mean())

        frequent_cats = sorted([
            cat for cat, n in value_counts.items()
            if n >= self.max_frequent
        ])

        rare_cats = [
            cat for cat, n in value_counts.items()
            if n < self.max_frequent
        ]

        self.onehot_encoder_ = {
            cat: i for i, cat in enumerate(frequent_cats)
        }

        self.target_encoder_ = {"__default__": global_mean}

        y_series = pd.Series(y_numpy)
        col_pd = pd.Series(_to_numpy_1d(col))

        for cat in rare_cats:
            mask = col_pd == cat

            if mask.any():
                self.target_encoder_[cat] = float(y_series[mask].mean())
            else:
                self.target_encoder_[cat] = global_mean

        self._col_name = col_name

        self.all_outputs_ = (
            [f"{col_name}__{cat}" for cat in frequent_cats]
            + [f"{col_name}__rare_target"]
        )

        return self

    def transform(self, col):
        col_name = self._col_name

        frequent_cats = sorted(self.onehot_encoder_.keys())
        frequent_set = set(frequent_cats)

        default_val = self.target_encoder_.get("__default__", 0.0)

        data_dict = {}

        for cat in frequent_cats:
            data_dict[f"{col_name}__{cat}"] = _make_ohe_col(col, cat)

        data_dict[f"{col_name}__rare_target"] = _make_target_col(
            col,
            self.target_encoder_,
            frequent_set,
            default_val,
        )

        return _make_dataframe(col, data_dict)

    def fit_transform(self, col, y):
        return self.fit(col, y).transform(col)