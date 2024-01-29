from .. import _dataframe as sbd
from ._base import Selector
from ._utils import list_difference


class CardinalityBelow(Selector):
    def __init__(self, threshold):
        self.threshold = threshold

    def select(self, df, ignore=()):
        all_selected = []
        for col_name in list_difference(sbd.column_names(df), ignore):
            column = sbd.col(df, col_name)
            try:
                n_unique = sbd.n_unique(column)
            except Exception:
                # n_unique can fail for example for polars columns with dtype Object
                pass
            else:
                if n_unique < self.threshold:
                    all_selected.append(col_name)
        return all_selected

    def __repr__(self):
        return f"cardinality_below({self.threshold})"


def cardinality_below(threshold):
    return CardinalityBelow(threshold)
