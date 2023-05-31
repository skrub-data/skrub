"""
Implements the FeatureAugmenter, a that allows chaining multiple fuzzy joins
on a table.
"""

from typing import List, Literal, Tuple

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from skrub._fuzzy_join import fuzzy_join


class FeatureAugmenter(BaseEstimator, TransformerMixin):
    """Augment a main table by automatically joining multiple auxiliary tables on it.

    Given a list of tables and key column names,
    fuzzy join them to the main table.

    The principle is as follows:

    1. The main table and the key column name are provided at initialisation.
    2. The auxiliary tables are provided for fitting, and will be joined
       sequentially when :func:`~FeatureAugmenter.transform` is called.

    It is advised to use hyperparameter tuning tools such as
    :class:`~sklearn.model_selection.GridSearchCV` to determine the best
    `match_score` parameter, as this can significantly improve your results.
    (see example 'Fuzzy joining dirty tables with the FeatureAugmenter'
    for an illustration)

    Parameters
    ----------
    tables : list of 2-tuples of (:obj:`~pandas.DataFrame`, str)
        List of (table, column name) tuples, the tables to join.
    main_key : str
        The key column name in the main table (passed during fit) on which
        the join will be performed.
    match_score : float, default=0
        Distance score between the closest matches that will be accepted.
        In a [0, 1] interval. 1 means that only a perfect match will be
        accepted, and zero means that the closest match will be accepted,
        no matter how distant.
        For numerical joins, this defines the maximum Euclidean distance
        between the matches.
    analyzer : {'word', 'char', 'char_wb'}, default=`char_wb`
        Analyzer parameter for the
        :obj:`~sklearn.feature_extraction.text.CountVectorizer` used for
        the string similarities.
        Describes whether the matrix `V` to factorize should be made of
        word counts or character n-gram counts.
        Option `char_wb` creates character n-grams only from text inside word
        boundaries; n-grams at the edges of words are padded with space.
    ngram_range : 2-tuple of int, default=(2, 4)
        The lower and upper boundaries of the range of n-values for different
         n-grams used in the string similarity. All values of `n` such
         that ``min_n <= n <= max_n`` will be used.

    See Also
    --------
    :func:`skrub.fuzzy_join` :
        Join two tables (dataframes) based on approximate column matching.

    :func:`skrub.datasets.get_ken_embeddings` :
        Download vector embeddings for many common entities (cities,
        places, people...).

    Examples
    --------
    >>> X = pd.DataFrame(['France', 'Germany', 'Italy'],
                         columns=['Country'])
    >>> X
    Country
    0   France
    1  Germany
    2    Italy

    >>> aux_table_1 = pd.DataFrame([['Germany', 84_000_000],
                                    ['France', 68_000_000],
                                    ['Italy', 59_000_000]],
                                    columns=['Country', 'Population'])
    >>> aux_table_1
       Country  Population
    0  Germany    84000000
    1   France    68000000
    2    Italy    59000000

    >>> aux_table_2 = pd.DataFrame([['French Republic', 2937],
                                    ['Italy', 2099],
                                    ['Germany', 4223],
                                    ['UK', 3186]],
                                    columns=['Country name', 'GDP (billion)'])
    >>> aux_table_2
        Country name  GDP (billion)
    0   French Republic      2937
    1        Italy           2099
    2      Germany           4223
    3           UK           3186

    >>> aux_table_3 = pd.DataFrame([['France', 'Paris'],
                                    ['Italia', 'Rome'],
                                    ['Germany', 'Berlin']],
                                    columns=['Countries', 'Capital'])
    >>> aux_table_3
      Countries Capital
    0    France   Paris
    1     Italia   Rome
    2   Germany  Berlin

    >>> aux_tables = [(aux_table_1, "Country"),
                      (aux_table_2, "Country name"),
                      (aux_table_3, "Countries")]

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

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureAugmenter":
        """Fit the instance to the main table.

        In practice, just checks if the key columns in X,
        the main table, and in the auxiliary tables exist.

        Parameters
        ----------
        X : :obj:`~pandas.DataFrame`, shape [n_samples, n_features]
            The main table, to be joined to the auxiliary ones.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        :obj:`FeatureAugmenter`
            Fitted :class:`FeatureAugmenter` instance (self).
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

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Transform `X` using the specified encoding scheme.

        Parameters
        ----------
        X : :obj:`~pandas.DataFrame`, shape [n_samples, n_features]
            The main table, to be joined to the auxiliary ones.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        :obj:`~pandas.DataFrame`
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
