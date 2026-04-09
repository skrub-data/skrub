from sklearn.base import TransformerMixin

from . import DropCols, _column_associations, _config
from . import _dataframe as sbd

_SUBSAMPLE_SIZE = 3000


class DropSimilar(TransformerMixin):
    """Drop columns found too redundant to the rest of the dataframe,
    according to association defined by Cramér's V.

    This is done by computing Cramér's V between every possible two columns,
    and sorting these couples in descending order. Then, for every association above
    a preestablished threshold:
    - If one of the two columns has already been dropped, nothing happens
    - Otherwise, the column that has the highest average association score with
    every other column is dropped.

    Parameters
    ----------
    threshold : float, default=0.8
        If True, drop the column if it contains only one unique value. Missing values
        count as one additional distinct value.

    Attributes
    ----------
    to_drop : list
        The names of columns evaluated for removal

    See Also
    --------
    DropUninformative :
        Drops columns for which various other criteria indicate that they contain
        little to no information (amount of nulls, of distinct values...)

    Cleaner :
        Runs several checks to sanitize a dataframe, including converting columns
        to standard formats or dropping certain columns.
    """

    def __init__(self, threshold=0.8):
        self.threshold = threshold
        self.to_drop = []
        self._dropper = DropCols([])

    def associations_as_report(self, X):
        df = sbd.sample(
            X,
            n=min(sbd.shape(X)[0], _SUBSAMPLE_SIZE),
            seed=_config.get_config()["subsampling_seed"],
        )

        return _column_associations.column_associations(df)

    def fit_transform(self, X, y=None):
        association_df = _column_associations.column_associations(X)

        averages = {
            col: association_df[association_df["left_column_name"] == col]
            for col in X.columns()
        }
        max_associations = association_df[association_df["cramer_v"] > self.threshold]

        self.to_drop = []

        for i in range(len(max_associations)):
            left, right = (
                max_associations["left_column_name"][i],
                max_associations["right_column_name"][i],
            )
            if left not in self.to_drop and right not in self.to_drop:
                if averages[left] > averages[right]:
                    self.to_drop.append(left)
                else:
                    self.to_drop.append(right)

        self._dropper = DropCols(self.to_drop)
        return self._dropper.fit_transform(X, y)

    def transform(self, X):
        return self._dropper.transform(X)
