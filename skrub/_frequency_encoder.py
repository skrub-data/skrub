"""
Implements the Frequency Encoder, a transformer that allows
encoding a feature using it's frequency.
"""
import pandas as pd

from ._dispatch import dispatch
from ._on_each_column import SingleColumnTransformer


@dispatch
def _cut(col):
    raise NotImplementedError()


@_cut.specialize("pandas", argument_type="Column")
def _cut_pandas(col):
    # TODO
    pass


@_cut.specialize("polars", argument_type="Column")
def _cut_polars(col):
    # TODO
    pass


class FrequencyEncoder(SingleColumnTransformer):
    def __init__(self, column, bins):
        self.column = column
        self.bins = bins
        self.uniques_to_map = None

    def fit(self, X, y=None):
        del y

        value_counts_series = X[self.column].value_counts()
        self.uniques_to_map = pd.cut(value_counts_series, self.bins, right=False)

        return self

    def transform(self, X):
        return X[self.column].map(self.uniques_to_map)
