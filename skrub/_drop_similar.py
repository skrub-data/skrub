import polars as pl
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_is_fitted

from skrub._dataframe._common import raise_dispatch_unregistered_type

from . import DropCols, _column_associations
from ._dispatch import dispatch


@dispatch
def _filter_associations(obj):
    raise_dispatch_unregistered_type(obj, kind="Series")


@_filter_associations.specialize("pandas")
def _filter_associations_pandas(obj, threshold):
    return obj[obj["cramer_v"] > threshold]


@_filter_associations.specialize("polars")
def _filter_associations_polars(obj, threshold):
    return obj.filter(pl.col("cramer_v") > threshold)


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

    def fit_transform(self, X, y=None):
        # check that the threshold is correct
        if not (0 <= self.threshold <= 1):
            raise ValueError("Threshold must be between 0 and 1")

        self.to_drop_ = []

        association_df = _column_associations.column_associations(X)

        pairs_to_drop = _filter_associations(association_df, self.threshold)

        self.to_drop_.extend(pairs_to_drop["right_column_name"].unique())

        self._dropper = DropCols(self.to_drop_)

        return self._dropper.fit_transform(X, y)

    def transform(self, X):
        check_is_fitted()
        return self._dropper.transform(X)
