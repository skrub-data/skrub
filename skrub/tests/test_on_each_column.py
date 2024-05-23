import numpy as np
import pytest

from skrub._on_each_column import OnEachColumn, RejectColumn, SingleColumnTransformer


def test_single_column_transformer_wrapped_methods(df_module):
    class Dummy(SingleColumnTransformer):
        def fit_transform(self, column, y=None):
            return column

        def transform(self, column):
            return column

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


def test_single_column_transformer_misc():
    class Dummy(SingleColumnTransformer):
        """dummy transformer

        details
        """

    assert "``Dummy`` is a type of single-column" in Dummy.__doc__
    assert Dummy.__single_column_transformer__ is True

    class Dummy(SingleColumnTransformer):
        pass

    assert Dummy.__doc__ is None
