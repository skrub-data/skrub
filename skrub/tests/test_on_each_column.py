import re

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_index_equal
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from skrub import _dataframe as sbd
from skrub._apply_to_cols import ApplyToCols
from skrub._select_cols import Drop
from skrub._single_column_transformer import SingleColumnTransformer


class RenameB(SingleColumnTransformer):
    """Dummy transformer to check column name deduplication."""

    def fit_transform(self, column, y=None):
        return self.transform(column)

    def transform(self, column):
        return sbd.rename(column, "B")


class Mult(BaseEstimator):
    """Dummy to test the different kinds of output supported by ApplyToCols.

    Supported kinds of output are a single column, a list of columns, or a
    dataframe. This also checks that when the transformer is not a
    single-column transformer, X is passed by ApplyToCols as a dataframe
    containing a single column (not as a column).
    """

    def __init__(self, output_kind="single_column"):
        self.output_kind = output_kind

    def fit_transform(self, X, y):
        self.y_ = np.asarray(y)
        return self.transform(X)

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        assert sbd.is_dataframe(X)
        assert sbd.shape(X)[1] == 1
        col_name = sbd.column_names(X)[0]
        col = sbd.col(X, col_name)
        outputs = sbd.make_dataframe_like(
            X, {f"{col_name} * y": col * self.y_, f"{col_name} * 2.0": col * 2.0}
        )
        if self.output_kind == "dataframe":
            return outputs
        if self.output_kind == "column_list":
            return sbd.to_column_list(outputs)
        assert self.output_kind == "single_column", self.output_kind
        return sbd.col(outputs, sbd.column_names(outputs)[0])


def test_empty_selection(df_module):
    mapper = ApplyToCols(Drop(), ())
    out = mapper.fit_transform(df_module.example_dataframe)
    df_module.assert_frame_equal(out, df_module.example_dataframe)
    assert mapper.transformers_ == {}


def test_empty_output(df_module):
    mapper = ApplyToCols(Drop())
    out = mapper.fit_transform(df_module.example_dataframe)
    if df_module.name == "pandas":
        expected = df_module.empty_dataframe.set_axis(
            df_module.example_dataframe.index, axis="index"
        )
    else:
        expected = df_module.empty_dataframe
    df_module.assert_frame_equal(out, expected)
    assert list(mapper.transformers_.keys()) == sbd.column_names(
        df_module.example_dataframe
    )


def _to_XXX(names):
    """Mask out the random part of column names."""
    return [re.sub(r"__skrub_[0-9a-f]+__", "__skrub_XXX__", n) for n in names]


def test_column_renaming(df_module, use_fit_transform):
    df = mapper = out = out_names = None

    def fit_transform():
        nonlocal out, out_names
        if use_fit_transform:
            out = mapper.fit_transform(df)
        else:
            out = mapper.fit(df).transform(df)
        out_names = _to_XXX(sbd.column_names(out))

    df = df_module.make_dataframe({"A": [1], "B": [2]})

    mapper = ApplyToCols(RenameB())
    fit_transform()
    assert out_names == ["B__skrub_XXX__", "B"]

    mapper = ApplyToCols(RenameB(), cols=("A",), rename_columns="{}_out")
    fit_transform()
    assert out_names == ["B_out", "B"]

    mapper = ApplyToCols(
        RenameB(), cols=("A",), rename_columns="{}_out", keep_original=True
    )
    fit_transform()
    assert out_names == ["A", "B_out", "B"]


class NumpyOutput(BaseEstimator):
    def fit_transform(self, X, y=None):
        return np.ones(sbd.shape(X))


def test_wrong_transformer_output_type(pd_module):
    with pytest.raises(TypeError, match=".*fit_transform returned a result of type"):
        ApplyToCols(NumpyOutput()).fit_transform(pd_module.example_dataframe)


def test_set_output_failure(df_module):
    # check that set_output on the transformer is allowed to fail even if the
    # set_output method exists, as long as the transformer produces output of
    # the right type.
    X = df_module.make_dataframe(
        {"a 0": [1.0, 2.2], "a 1": [3.0, 4.4], "b": [5.0, 6.6]}
    )
    y = [0.0, 1.0]
    mapper = ApplyToCols(make_pipeline(Mult()))
    mapper.fit_transform(X, y)


class ResetsIndex(BaseEstimator):
    def fit_transform(self, X, y=None):
        return X.reset_index()

    def transform(self, X):
        return X.reset_index()


@pytest.mark.parametrize("cols", [(), ("a",), ("a", "b")])
def test_output_index(cols):
    df = pd.DataFrame({"a": [10, 20], "b": [1.1, 2.2]}, index=[-1, -2])
    transformer = ApplyToCols(ResetsIndex(), cols=cols)
    assert_index_equal(transformer.fit_transform(df).index, df.index)
    df = pd.DataFrame({"a": [10, 20], "b": [1.1, 2.2]}, index=[-10, 20])
    assert_index_equal(transformer.transform(df).index, df.index)


def test_set_output_polars(pl_module):
    # non-regression for an issue introduced in #973; set_output('polars')
    # would cause a failure in old scikit-learn versions.
    # see #1122 for details
    df = pl_module.make_dataframe({"x": ["a", "b", "c"]})
    ApplyToCols(OneHotEncoder(sparse_output=False)).fit_transform(df)
