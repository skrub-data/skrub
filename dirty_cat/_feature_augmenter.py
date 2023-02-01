"""
Transformer that allows multiple fuzzy joins to be performed on a table.
The principle is as follows:
  1. The main table and the key column name are provided at initialisation.
  2. The auxilliary tables are provided for fitting, and will be joined sequentially
  when the transform is called.
It is advised to use hyper-parameter tuning tools such as scikit-learn's
GridSearchCV to determine the best `match_score` parameter, as this can
significantly improve your results.
(see example 'Fuzzy joining dirty tables with the FeatureAugmenter' for an illustration)
For more information on how the join is performed, see fuzzy_join's documentation.
"""

from typing import List, Literal, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from dirty_cat._fuzzy_join import fuzzy_join


class FeatureAugmenter(BaseEstimator, TransformerMixin):
    """Transformer augmenting the number of features in a table by joining multiple tables.

    Given a list of tables and key column names,
    fuzzy join them to the main table.

    Parameters
    ----------
    tables : list of 2-tuples of (:class:`~pandas.DataFrame`, str)
        List of (table, column name) tuples
        specyfying the transformer objects to be applied.
        table: str
            Name of the table to be joined.
        column: str
            Name of table column to join on.
    main_key : str
        The key column name in the main table on which
        the join will be performed.
    match_score : float, default=0
        Distance score between the closest matches that will be accepted.
        In a [0, 1] interval. 1 means that only a perfect match will be
        accepted, and zero means that the closest match will be accepted,
        no matter how distant.
    analyzer : typing.Literal["word", "char", "char_wb"], default=`char_wb`
        Analyzer parameter for the CountVectorizer used for the string
        similarities.
        Options: {`word`, `char`, `char_wb`}, describing whether the matrix V
        to factorize should be made of word counts or character n-gram counts.
        Option `char_wb` creates character n-grams only from text inside word
        boundaries; n-grams at the edges of words are padded with space.
    ngram_range : tuple (min_n, max_n), default=(2, 4)
        The lower and upper boundary of the range of n-values for different
        n-grams used in the string similarity. All values of n such
        that min_n <= n <= max_n will be used.

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

    >>> aux_table_2 = pd.DataFrame([['French Republic', 2937], ['Italy', 2099], ['Germany', 4223], ['UK', 3186]], columns=['Country name', 'GDP (billion)']) # noqa
    >>> aux_table_2
        Country name  GDP (billion)
    0   French Republic      2937
    1        Italy           2099
    2      Germany           4223
    3           UK           3186

    >>> aux_table_3 = pd.DataFrame([['France', 'Paris'], ['Italia', 'Rome'], ['Germany', 'Berlin']], columns=['Countries', 'Capital']) # noqa
    >>> aux_table_3
      Countries Capital
    0    France   Paris
    1     Italia   Rome
    2   Germany  Berlin

    >>> aux_tables = [(aux_table_1, "Country"), (aux_table_2, "Country name"), (aux_table_3, "Countries")] # noqa

    >>> fa = FeatureAugmenter(tables=aux_tables, main_key='Country')

    >>> augmented_table = fa.fit_transform(X)
    >>> augmented_table
        Country Country_aux  Population Country name  GDP (billion) Countries Capital
    0   France      France    68000000  French Republic       2937    France   Paris
    1  Germany     Germany    84000000      Germany           4223   Germany  Berlin
    2    Italy       Italy    59000000        Italy           2099    Italia    Rome
    """

    def __init__(
        self,
        tables: List[Tuple[pd.DataFrame, str]],
        main_key: str,
        match_score: float = 0.0,
        analyzer: Literal["word", "char", "char_wb"] = "char_wb",
        ngram_range: Tuple[int, int] = (2, 4),
    ):
        self.tables = tables
        self.main_key = main_key
        self.match_score = match_score
        self.analyzer = analyzer
        self.ngram_range = ngram_range

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
                f"Got main_key={self.main_key!r}, but column not in {list(X.columns)}."
            )

        for pairs in self.tables:
            if pairs[1] not in pairs[0].columns:
                raise ValueError(
                    f"Got column key {pairs[1]!r}, "
                    f"but column not in {pairs[0].columns}."
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

        for pairs in self.tables:
            # TODO: Add an option to fuzzy_join on multiple columns at once
            # (will be if len(inter_col)!=0)
            aux_table = pairs[0]
            X = fuzzy_join(
                X,
                aux_table,
                left_on=self.main_key,
                right_on=pairs[1],
                match_score=self.match_score,
                analyzer=self.analyzer,
                ngram_range=self.ngram_range,
                suffixes=("", "_aux"),
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
