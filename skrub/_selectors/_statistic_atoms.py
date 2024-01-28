from .._dataframe import asdfapi, asnative
from ._base import Selector


class CardinalityBelow(Selector):
    def __init__(self, threshold):
        self.threshold = threshold

    def select(self, df):
        df = asdfapi(df)
        all_selected = []
        for col_name in df.column_names:
            column = df.col(col_name)
            if asnative(column.n_unique()) < self.threshold:
                all_selected.append(col_name)
        return all_selected

    def __repr__(self):
        return f"cardinality_below({self.threshold})"


def cardinality_below(threshold):
    return CardinalityBelow(threshold)
