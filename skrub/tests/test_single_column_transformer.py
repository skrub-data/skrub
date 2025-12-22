import numpy as np
import pytest
from sklearn.base import BaseEstimator

from skrub import _dataframe as sbd
from skrub import selectors as s
from skrub._apply_to_cols import ApplyToCols
from skrub._single_column_transformer import RejectColumn, SingleColumnTransformer


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


def test_single_column_transformer_all_outputs(df_module):
    class Dummy(SingleColumnTransformer):
        def fit(self, column, y=None):
            self.all_outputs_ = [sbd.name(column)]
            return column

    column = df_module.example_column

    transformer = Dummy()
    transformer.fit(column)

    assert transformer.get_feature_names_out() == [sbd.name(column)]


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


class SingleColMult(Mult):
    """Single-column transformer equivalent of Mult."""

    __single_column_transformer__ = True

    def transform(self, X):
        assert sbd.is_column(X)
        X = sbd.make_dataframe_like(X, [X])
        return super().transform(X)


@pytest.mark.parametrize("output_kind", ["single_column", "dataframe", "column_list"])
@pytest.mark.parametrize("transformer_class", [Mult, SingleColMult])
def test_single_column_transformer(
    df_module, output_kind, transformer_class, use_fit_transform
):
    mapper = ApplyToCols(transformer_class(output_kind), s.glob("a*"))
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


class Rejector(SingleColumnTransformer):
    """Dummy class to test ApplyToCols behavior when columns are rejected."""

    def fit_transform(self, column, y=None):
        if sbd.name(column) != "float-col":
            raise RejectColumn("only float-col is accepted")
        return self.transform(column)

    def transform(self, column):
        return column * 2.2


def test_allowed_column_rejections(df_module, use_fit_transform):
    df = df_module.example_dataframe
    mapper = ApplyToCols(Rejector(), allow_reject=True)
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
    mapper = ApplyToCols(Rejector())
    with pytest.raises(ValueError, match=".*failed on.*int-col"):
        mapper.fit(df)


class RejectInTransform(SingleColumnTransformer):
    def fit_transform(self, column, y=None):
        return column

    def transform(self, column):
        raise RejectColumn()


def test_rejection_forbidden_in_transform(df_module):
    df = df_module.example_dataframe
    mapper = ApplyToCols(RejectInTransform(), allow_reject=True)
    mapper.fit(df)
    with pytest.raises(ValueError, match=".*failed on.*int-col"):
        mapper.transform(df)


class RenameB(SingleColumnTransformer):
    """Dummy transformer to check column name deduplication."""

    def fit_transform(self, column, y=None):
        return self.transform(column)

    def transform(self, column):
        return sbd.rename(column, "B")
