import re

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_index_equal
from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline

from skrub import _dataframe as sbd
from skrub import _selectors as s
from skrub._on_each_column import OnEachColumn, RejectColumn, SingleColumnTransformer
from skrub._select_cols import Drop


@pytest.mark.parametrize("define_fit", [False, True])
def test_single_column_transformer_wrapped_methods(df_module, define_fit):
    class Dummy(SingleColumnTransformer):
        def fit_transform(self, column, y=None):
            return column

        def transform(self, column):
            return column

        if define_fit:

            def fit(self, column, y=None):
                return self

    col = df_module.example_column
    assert Dummy().fit_transform(col) is col
    assert Dummy().fit(col).transform(col) is col

    dummy = Dummy().fit(col)
    for method in "fit", "fit_transform", "transform":
        with pytest.raises(
            ValueError, match=r"``Dummy\..*`` should be passed a single column"
        ):
            getattr(dummy, method)(df_module.example_dataframe)

        with pytest.raises(
            ValueError, match=r"``Dummy\..*`` expects the first argument X"
        ):
            getattr(dummy, method)(np.ones((3,)))


@pytest.mark.parametrize(
    "docstring",
    [
        "dummy transformer\n\n    details\n",
        "\n    dummy transformer\n    summary\n\n    details",
        "summary",
        "\n    dummy transformer\n\ndetails\n   \n    more",
        "",
    ],
)
def test_single_column_transformer_docstring(docstring):
    class Dummy(SingleColumnTransformer):
        __doc__ = docstring

    assert "``Dummy`` is a type of single-column" in Dummy.__doc__

    class Dummy(SingleColumnTransformer):
        pass

    assert Dummy.__doc__ is None


def test_single_column_transformer_attribute():
    class Dummy(SingleColumnTransformer):
        pass

    assert Dummy.__single_column_transformer__ is True


class Mult(BaseEstimator):
    """Dummy to test the different kinds of output supported by OnEachColumn.

    Supported kinds of output are a single column, a list of columns, or a
    dataframe. This also checks that when the transformer is not a
    single-column transformer, X is passed by OnEachColumn as a dataframe
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


class SingleColMult(Mult):
    """Single-column transformer equivalent of Mult."""

    __single_column_transformer__ = True

    def transform(self, X):
        assert sbd.is_column(X)
        X = sbd.make_dataframe_like(X, [X])
        return super().transform(X)


@pytest.mark.parametrize("output_kind", ["single_column", "dataframe", "column_list"])
@pytest.mark.parametrize("transformer_class", [Mult, SingleColMult])
def test_on_each_column(df_module, output_kind, transformer_class, use_fit_transform):
    mapper = OnEachColumn(transformer_class(output_kind), s.glob("a*"))
    X = df_module.make_dataframe(
        {"a 0": [1.0, 2.2], "a 1": [3.0, 4.4], "b": [5.0, 6.6]}
    )
    y = [0.0, 1.0]
    if use_fit_transform:
        out = mapper.fit_transform(X, y)
    else:
        out = mapper.fit(X, y).transform(X)
    expected_data = {
        "a 0 * y": [0.0, 2.2],
        "a 0 * 2.0": [2.0, 4.4],
        "a 1 * y": [0.0, 4.4],
        "a 1 * 2.0": [6.0, 8.8],
        "b": [5.0, 6.6],
    }
    if output_kind == "single_column":
        expected_data.pop("a 0 * 2.0")
        expected_data.pop("a 1 * 2.0")
    expected = df_module.make_dataframe(expected_data)
    df_module.assert_frame_equal(out, expected)
    assert mapper.get_feature_names_out() == list(expected_data.keys())


def test_empty_selection(df_module):
    mapper = OnEachColumn(Drop(), ())
    out = mapper.fit_transform(df_module.example_dataframe)
    df_module.assert_frame_equal(out, df_module.example_dataframe)
    assert mapper.transformers_ == {}


def test_empty_output(df_module):
    mapper = OnEachColumn(Drop())
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


class Rejector(SingleColumnTransformer):
    """Dummy class to test OnEachColumn behavior when columns are rejected."""

    def fit_transform(self, column, y=None):
        if sbd.name(column) != "float-col":
            raise RejectColumn("only float-col is accepted")
        return self.transform(column)

    def transform(self, column):
        return column * 2.2


def test_allowed_column_rejections(df_module, use_fit_transform):
    df = df_module.example_dataframe
    mapper = OnEachColumn(Rejector(), allow_reject=True)
    if use_fit_transform:
        out = mapper.fit_transform(df)
    else:
        out = mapper.fit(df).transform(df)
    assert sbd.column_names(out) == sbd.column_names(df)
    df_module.assert_column_equal(
        sbd.col(out, "float-col"), sbd.col(df, "float-col") * 2.2
    )
    for col_name in sbd.column_names(df):
        if col_name != "float-col":
            df_module.assert_column_equal(sbd.col(out, col_name), sbd.col(df, col_name))


def test_forbidden_column_rejections(df_module):
    df = df_module.example_dataframe
    mapper = OnEachColumn(Rejector())
    with pytest.raises(ValueError, match=".*failed on.*int-col"):
        mapper.fit(df)


class RejectInTransform(SingleColumnTransformer):
    def fit_transform(self, column, y=None):
        return column

    def transform(self, column):
        raise RejectColumn()


def test_rejection_forbidden_in_transform(df_module):
    df = df_module.example_dataframe
    mapper = OnEachColumn(RejectInTransform(), allow_reject=True)
    mapper.fit(df)
    with pytest.raises(ValueError, match=".*failed on.*int-col"):
        mapper.transform(df)


class RenameB(SingleColumnTransformer):
    """Dummy transformer to check column name deduplication."""

    def fit_transform(self, column, y=None):
        return self.transform(column)

    def transform(self, column):
        return sbd.rename(column, "B")


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

    mapper = OnEachColumn(RenameB())
    fit_transform()
    assert out_names == ["B__skrub_XXX__", "B"]

    mapper = OnEachColumn(RenameB(), cols=("A",), rename_columns="{}_out")
    fit_transform()
    assert out_names == ["B_out", "B"]

    mapper = OnEachColumn(
        RenameB(), cols=("A",), rename_columns="{}_out", keep_original=True
    )
    fit_transform()
    assert out_names == ["A", "B_out", "B"]


class NumpyOutput(BaseEstimator):
    def fit_transform(self, X, y=None):
        return np.ones(sbd.shape(X))


def test_wrong_transformer_output_type(all_dataframe_modules):
    with pytest.raises(TypeError, match=".*fit_transform returned a result of type"):
        OnEachColumn(NumpyOutput()).fit_transform(
            all_dataframe_modules["pandas-numpy-dtypes"].example_dataframe
        )


def test_set_output_failure(df_module):
    # check that set_output on the transformer is allowed to fail even if the
    # set_output method exists, as long as the transformer produces output of
    # the right type.
    X = df_module.make_dataframe(
        {"a 0": [1.0, 2.2], "a 1": [3.0, 4.4], "b": [5.0, 6.6]}
    )
    y = [0.0, 1.0]
    mapper = OnEachColumn(make_pipeline(Mult()))
    mapper.fit_transform(X, y)


class ResetsIndex(BaseEstimator):
    def fit_transform(self, X, y=None):
        return X.reset_index()

    def transform(self, X):
        return X.reset_index()


@pytest.mark.parametrize("cols", [(), ("a",), ("a", "b")])
def test_output_index(cols):
    df = pd.DataFrame({"a": [10, 20], "b": [1.1, 2.2]}, index=[-1, -2])
    transformer = OnEachColumn(ResetsIndex(), cols=cols)
    assert_index_equal(transformer.fit_transform(df).index, df.index)
    df = pd.DataFrame({"a": [10, 20], "b": [1.1, 2.2]}, index=[-10, 20])
    assert_index_equal(transformer.transform(df).index, df.index)
