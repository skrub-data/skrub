"""
A transformer that removes columns from a dataframe if they are too closely
correlated to at least one other.
"""

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
    threshold : float, default=1
        The Cramér association score value above which to start dropping columns.

    Attributes
    ----------
    to_drop_ : list
        The names of columns evaluated for removal

    all_outputs_ : list
        The names of columns that the transformer keeps

    column_associations_ : dataframe
        A dataframe with columns `left_column_name', 'right_column_name' and 'cramer_v'
        listing association scores between every pair of columns.

    See Also
    --------
    DropUninformative :
        Drops a single column if certain criteria indicate that it contains
        little to no information (amount of nulls, of distinct values...)

    Cleaner :
        Runs several checks to sanitize a dataframe, including converting columns
        to standard formats or dropping certain columns.

    Examples
    --------
    >>> from skrub import DropSimilar
    >>> from skrub.datasets import toy_cities
    >>> df = toy_cities(size=5000, n_metrics=0)
    >>> df.head() # doctest: +SKIP
              uid   cities  encoded_cities               start                 end
    0  SHAoqcdajQ  Vilnius            17.0 2004-09-02 03:22:56 2014-06-26 11:36:43
    1  HVAFYLGCDW      NaN             NaN 1979-10-22 01:43:56 1987-10-15 06:30:05
    2  oQIauSCbNL     Rome            13.0 1986-08-09 19:01:10 2002-04-06 03:56:09
    3  SjeSbCepzv  Vilnius            17.0 2008-11-26 15:57:13 2021-11-16 23:16:13
    4  ubagaIBHnG   London             8.0 1982-09-13 20:54:54 2000-12-02 07:36:41

    >>> ds = DropSimilar(threshold=0.8)
    >>> clean_df = ds.fit_transform(df)

    `ds` has now removed a column for each pair with association above 0.8.
    These associations are stored in the `table_associations_` attribute:

    >>> ds.table_associations_.head() # doctest: +SKIP
      left_column_name right_column_name  cramer_v
    0           cities    encoded_cities  1.000000
    1              uid            cities  0.052979
    2              uid    encoded_cities  0.052979
    3   encoded_cities               end  0.046455
    4           cities               end  0.046455

    A single pair is above the threshold, `cities` and `encoded_cities`,
    with an association score of 1. Since one is an encoding of the other,
    this is to be expected.
    Therefore, one of these two has been marked as dropped by `ds`:

    >>> ds.to_drop_
    ['encoded_cities']

    This leaves us with the shortened dataframe:

    >>> clean_df.head() # doctest: +SKIP
              uid   cities               start                 end
    0  SHAoqcdajQ  Vilnius 2004-09-02 03:22:56 2014-06-26 11:36:43
    1  HVAFYLGCDW      NaN 1979-10-22 01:43:56 1987-10-15 06:30:05
    2  oQIauSCbNL     Rome 1986-08-09 19:01:10 2002-04-06 03:56:09
    3  SjeSbCepzv  Vilnius 2008-11-26 15:57:13 2021-11-16 23:16:13
    4  ubagaIBHnG   London 1982-09-13 20:54:54 2000-12-02 07:36:41
    """  # noqa: E501

    def __init__(self, threshold=1):
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
            isinstance(self.threshold, numbers.Number)
        ):
            raise ValueError(f"Threshold must be a number, got {self.threshold}")
        elif not 0 <= self.threshold <= 1:
            raise ValueError(
                f"Threshold must be a number between 0 and 1, got {self.threshold!r}."
            )

        if sbd.is_polars(X):
            try:
                import pyarrow  # noqa F401
            except ImportError:
                raise ImportError(
                    "DropSimilar requires the Pyarrow package to run on Polars"
                    " dataframes."
                )

        association_df = column_associations(X)
        self.column_associations_ = s.select(
            association_df, ["left_column_name", "right_column_name", "cramer_v"]
        )

        pairs_to_drop = _filter_associations(self.column_associations_, self.threshold)

        self.to_drop_ = sbd.to_list(sbd.unique(pairs_to_drop["right_column_name"]))

        self._dropper = DropCols(self.to_drop_)
        new_X = self._dropper.fit_transform(X, y)

        self.all_outputs_ = self._dropper.kept_cols_

        return new_X

    def transform(self, X):
        check_is_fitted(self)
        return self._dropper.transform(X)
