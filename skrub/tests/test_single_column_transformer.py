import numpy as np
import pytest
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from skrub import _dataframe as sbd
from skrub._single_column_transformer import (
    SingleColumnTransformer,
    is_single_column_transformer,
)


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


def test_is_single_column_transformer():
    class S:
        __single_column_transformer__ = True

    for t in [
        S(),
        make_pipeline(S(), StandardScaler()),
        make_pipeline(make_pipeline(S(), StandardScaler()), StandardScaler()),
    ]:
        assert is_single_column_transformer(t)
    for t in [
        StandardScaler(),
        None,
        Pipeline(()),
        make_pipeline(StandardScaler(), S()),
    ]:
        assert not is_single_column_transformer(t)
