try:
    import polars as pl
except ImportError:
    pass
import numbers

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from . import selectors as s
from ._column_associations import column_associations
from ._dataframe._common import raise_dispatch_unregistered_type
from ._dispatch import dispatch
from ._select_cols import DropCols


@dispatch
def _filter_associations(obj):
    raise_dispatch_unregistered_type(obj, kind="Series")


@_filter_associations.specialize("pandas")
def _filter_associations_pandas(obj, threshold):
    return obj[obj["cramer_v"] >= threshold]


@_filter_associations.specialize("polars")
def _filter_associations_polars(obj, threshold):
    return obj.filter(pl.col("cramer_v") >= threshold)


class DropSimilar(TransformerMixin, BaseEstimator):
    """Drop columns found too redundant to the rest of the dataframe,
    according to association defined by Cramér's V.

    This is done by computing Cramér's V between every possible two columns,
    and sorting these couples in descending order. Then, for every association above
    the given threshold, one of the two columns is dropped.

    Parameters
    ----------
    threshold : float, default=0.8
        The Cramér association score value above which to start dropping columns.

    Attributes
    ----------
    to_drop_ : list
        The names of columns evaluated for removal

    all_outputs_ : list
        The names of columns that the transformer keeps

    table_associations_ : dataframe
        A dataframe with columns `left_column_name', 'right_column_name' and 'cramer_v'
        listing association scores between every pair of columns.

    See Also
    --------
    DropUninformative :
        Drops columns for which various other criteria indicate that they contain
        little to no information (amount of nulls, of distinct values...)

    Cleaner :
        Runs several checks to sanitize a dataframe, including converting columns
        to standard formats or dropping certain columns.

    Examples
    --------
    """

    def __init__(self, threshold=0.8):
        self.threshold = threshold

    def get_feature_names_out(self):
        check_is_fitted(self)
        return self.all_outputs_

    def fit(self, X, y=None):
        self.fit_transform(X, y=y)
        return self

    def fit_transform(self, X, y=None):
        # check that the threshold is correct
        if isinstance(self.threshold, bool) or not (
            isinstance(self.threshold, numbers.Real) and 0 <= self.threshold <= 1
        ):
            raise ValueError(
                f"Threshold must be a number between 0 and 1, got {self.threshold!r}."
            )

        if isinstance(X, pl.DataFrame):
            try:
                import pyarrow  # noqa F401
            except ImportError:
                raise ImportError(
                    "DropSimilar requires the Pyarrow package to run on Polars"
                    " dataframes."
                )

        self.to_drop_ = []

        association_df = column_associations(X)
        self.table_associations_ = s.select(
            association_df, ["left_column_name", "right_column_name", "cramer_v"]
        )

        pairs_to_drop = _filter_associations(self.table_associations_, self.threshold)

        self.to_drop_.extend(
            sbd.to_list(sbd.unique(pairs_to_drop["right_column_name"]))
        )

        self._dropper = DropCols(self.to_drop_)
        new_X = self._dropper.fit_transform(X, y)

        self.all_outputs_ = self._dropper.kept_cols_

        return new_X

    def transform(self, X):
        check_is_fitted(self)
        return self._dropper.transform(X)
