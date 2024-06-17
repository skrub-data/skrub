import re

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_index_equal
from sklearn.base import BaseEstimator
from sklearn.preprocessing import FunctionTransformer

from skrub import SelectCols
from skrub import _dataframe as sbd
from skrub import _selectors as s
from skrub._on_subframe import OnSubFrame


class Dummy(BaseEstimator):
    def fit_transform(self, X, y):
        self.y_ = np.asarray(y)
        return self.transform(X)

    def transform(self, X):
        result = {}
        for name in sbd.column_names(X):
            col = sbd.col(X, name)
            result[f"{name} * y"] = col * self.y_
            result[f"{name} * 2.0"] = col * 2.0
        return sbd.make_dataframe_like(X, result)

    def fit(self, df, y):
        self.fit_transform(df, y)
        return self


def test_on_subframe(df_module, use_fit_transform):
    X = df_module.make_dataframe({"a": [1.0, 2.2], "b": [3.0, 4.4], "c": [5.0, 6.6]})
    y = [0.0, 1.0]
    transformer = OnSubFrame(Dummy(), ["a", "c"])
    if use_fit_transform:
        out = transformer.fit_transform(X, y)
    else:
        out = transformer.fit(X, y).transform(X)
    expected_data = {
        "b": [3.0, 4.4],
        "a * y": [0.0, 2.2],
        "a * 2.0": [2.0, 4.4],
        "c * y": [0.0, 6.6],
        "c * 2.0": [10.0, 13.2],
    }
    expected = df_module.make_dataframe(expected_data)
    df_module.assert_frame_equal(out, expected)
    assert transformer.get_feature_names_out() == list(expected_data.keys())


def test_empty_selection(df_module, use_fit_transform):
    df = df_module.example_dataframe
    transformer = OnSubFrame(Dummy(), ())
    if use_fit_transform:
        out = transformer.fit_transform(df)
    else:
        out = transformer.fit(df).transform(df)
    df_module.assert_frame_equal(out, df)


def test_empty_output(df_module, use_fit_transform):
    df = df_module.example_dataframe
    transformer = OnSubFrame(SelectCols(()))
    if use_fit_transform:
        out = transformer.fit_transform(df)
    else:
        out = transformer.fit(df).transform(df)
    df_module.assert_frame_equal(out, s.select(df, ()))


def _to_XXX(names):
    """Mask out the random part of column names."""
    return [re.sub(r"__skrub_[0-9a-f]+__", "__skrub_XXX__", n) for n in names]


def test_keep_original(df_module, use_fit_transform):
    df = df_module.make_dataframe({"A": [1], "B": [2]})
    transformer = OnSubFrame(FunctionTransformer(), keep_original=True)

    if use_fit_transform:
        out = transformer.fit_transform(df)
    else:
        out = transformer.fit(df).transform(df)

    out_names = _to_XXX(sbd.column_names(out))
    assert out_names == ["A", "B", "A__skrub_XXX__", "B__skrub_XXX__"]


class NumpyOutput(BaseEstimator):
    def fit_transform(self, X, y=None):
        return np.ones(sbd.shape(X))


def test_wrong_transformer_output_type(all_dataframe_modules):
    with pytest.raises(TypeError, match=".*fit_transform returned a result of type"):
        OnSubFrame(NumpyOutput()).fit_transform(
            all_dataframe_modules["pandas-numpy-dtypes"].example_dataframe
        )


class ResetsIndex(BaseEstimator):
    def fit_transform(self, X, y=None):
        return X.reset_index()

    def transform(self, X):
        return X.reset_index()


@pytest.mark.parametrize("cols", [(), ("a",), ("a", "b")])
def test_output_index(cols):
    df = pd.DataFrame({"a": [10, 20], "b": [1.1, 2.2]}, index=[-1, -2])
    transformer = OnSubFrame(ResetsIndex(), cols=cols)
    assert_index_equal(transformer.fit_transform(df).index, df.index)
    df = pd.DataFrame({"a": [10, 20], "b": [1.1, 2.2]}, index=[-10, 20])
    assert_index_equal(transformer.transform(df).index, df.index)
