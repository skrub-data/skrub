from .. import _dataframe as sbd
from ._base import Selector


class CardinalityBelow(Selector):
    def __init__(self, threshold):
        self.threshold = threshold

    def select(self, df):
        all_selected = []
        for col_name in sbd.column_names(df):
            column = sbd.col(df, col_name)
            if sbd.n_unique(column) < self.threshold:
                all_selected.append(col_name)
        return all_selected

    def __repr__(self):
        return f"cardinality_below({self.threshold})"


def cardinality_below(threshold):
    return CardinalityBelow(threshold)
