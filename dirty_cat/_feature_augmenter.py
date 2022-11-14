import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from dirty_cat._fuzzy_join import fuzzy_join


class FeatureAugmenter(BaseEstimator, TransformerMixin):
    """Transformer that helps augment the number of features in a table.

    Given a dictionnary of tables and key column names,
    fuzzy join them to the main table.

    Parameters
    ----------
    tables : dict
        A dictionnary containing the key column
        names and auxilliary tables that will be joined.
    main_key : str
        The key column name in the main table on which
        the join will be performed.
    match_score : float, default=0
        Distance score between the closest matches that will be accepted.
        In a [0, 1] interval. Closer to 1 means the matches need to be very
        close to be accepted, and closer to 0 that a bigger matching distance
        is tolerated. Equivalent to fuzzy_join's match_score.


    Notes
    -----
    The `tables` parameter is a dictionnary. This implies
    that key column names must be different.

    Examples
    --------
    >>> X = pd.DataFrame(['France', 'Germany', 'Italy'], columns=['Country'])
    >>> X
    Country
    0   France
    1  Germany
    2    Italy

    >>> aux_table_1 = pd.DataFrame([['Germany', 84_000_000], ['France', 68_000_000], ['Italy', 59_000_000]], columns=['Country', 'Population']) # noqa
    >>> aux_table_1
       Country  Population
    0  Germany    84000000
    1   France    68000000
    2    Italy    59000000

    >>> aux_table_2 = pd.DataFrame([['France', 2937], ['Italy', 2099], ['Germany', 4223]], columns=['Country name', 'GDP (billion)']) # noqa
    >>> aux_table_2
        Country name  GDP (billion)
    0       France           2937
    1        Italy           2099
    2      Germany           4223

    >>> aux_table_3 = pd.DataFrame([['France', 'Paris'], ['Italy', 'Rome'], ['Germany', 'Berlin']], columns=['Countries', 'Capital']) # noqa
    >>> aux_table_3
      Countries Capital
    0    France   Paris
    1     Italy    Rome
    2   Germany  Berlin

    >>> aux_dict = {"Country": aux_table_1, "Country name": aux_table_2, "Countries": aux_table_3} # noqa

    >>> fa = FeatureAugmenter(tables=aux_dict, main_key='Country')

    >>> augmented_table = fa.fit_transform(X)
    >>> augmented_table
        Country Country_aux  Population Country name  GDP (billion) Countries Capital
    0   France      France    68000000       France           2937    France   Paris
    1  Germany     Germany    84000000      Germany           4223   Germany  Berlin
    2    Italy       Italy    59000000        Italy           2099     Italy    Rome
    """

    def __init__(
        self,
        tables: dict,
        main_key: str,
        match_score=0,
    ):
        self.tables = tables
        self.main_key = main_key
        self.match_score = match_score

    def fit(self, X, y=None) -> "FeatureAugmenter":
        """Fit the Feature Augmenter to the main table.

        In practice, just checks if the key columns in X,
        the main table, and in the auxilliary tables exist.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The main table, to be joined to the
            auxilliary ones.
        y : array-like of shape (n_samples,...), default=None
            Targets for supervised learning.

        Returns
        -------
        FeatureAugmenter
            Fitted FeatureAugmenter instance.
        """

        if self.main_key not in X.columns:
            raise ValueError(
                f"Got main_key={self.main_key!r}, but column missing in the main table."
            )

        for key in self.tables:
            if key not in self.tables[key].columns:
                raise ValueError(
                    f"Got column key {key!r}, "
                    "but column missing in the auxilliary table."
                )
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """Transform X using the specified encoding scheme.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The main table, to be joined to the
            auxilliary ones.
        y : array-like of shape (n_samples,...), default=None
            Targets for supervised learning.

        Returns
        -------
        DataFrame
            The final joined table.
        """

        for key in self.tables:
            # TODO: Add an option to fuzzy_join on multiple columns at once
            # (will be if len(inter_col)!=0)

            aux_table = self.tables[key]
            X = fuzzy_join(
                X,
                aux_table,
                left_on=self.main_key,
                right_on=key,
                suffixes=("", "_aux"),
                match_score=self.match_score,
            )

        return X

    def fit_transform(self, X, y=None) -> pd.DataFrame:
        """
        Fit the FeatureAugmenter to X, then transforms it.

        Parameters
        ----------
        X : DataFrame, shape [n_samples, n_features]
            The main table, to be joined to the
            auxilliary ones.
        y : array-like of shape (n_samples,...), default=None
            Targets for supervised learning.

        Returns
        -------
        DataFrame
            The final joined table.
        """

        return self.fit(X).transform(X)
