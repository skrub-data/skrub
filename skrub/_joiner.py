"""
Implements the Joiner, a transformer that allows
multiple fuzzy joins on a table.
"""
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skrub import _join_utils
from skrub._datetime_encoder import DatetimeEncoder
from skrub._matching import TargetNeighborhood

DEFAULT_MATCHING = TargetNeighborhood()


def _make_vectorizer(table):
    transformers = [
        (TfidfVectorizer(), c)
        for c in table.select_dtypes(include=["string", "category", "object"]).columns
    ]
    transformers += [
        ("passthrough", table.select_dtypes(include="number").columns),
        (
            DatetimeEncoder(resolution=None, add_total_seconds=True),
            table.select_dtypes("datetime").columns,
        ),
    ]
    return make_pipeline(make_column_transformer(*transformers), StandardScaler())


class Joiner(TransformerMixin, BaseEstimator):
    """Augment a main table by fuzzy joining an auxiliary table to it.

    Given an auxiliary table and matching column names, fuzzy join it to the main
    table.
    The principle is as follows:

    1. The auxiliary table and the matching column names are provided at initialisation.
    2. The main table is provided for fitting, and will be joined
       when ``Joiner.transform`` is called.

    It is advised to use hyperparameter tuning tools such as GridSearchCV
    to determine the best `match_score` parameter, as this can significantly
    improve your results.
    (see example 'Fuzzy joining dirty tables with the Joiner'
    for an illustration)

    Parameters
    ----------
    aux_table : :obj:`~pandas.DataFrame`
        The auxiliary table, which will be fuzzy-joined to the main table when
        calling ``transform``.
    main_key : str or list of str, default=None
        The column names in the main table on which the join will be performed.
        Can be a string if joining on a single column.
        If ``None``, `aux_key` must also be ``None`` and `key` must be provided.
    aux_key : str or list of str, default=None
        The column names in the auxiliary table on which the join will
        be performed. Can be a string if joining on a single column.
        If ``None``, `main_key` must also be ``None`` and `key` must be provided.
    key : str or list of str, default=None
        The column names to use for both ``main_key`` and ``aux_key`` when they
        are the same. Provide either ``key`` or both ``main_key`` and ``aux_key``.
    suffix : str, default=""
        Suffix to append to the ``aux_table``'s column names. You can use it
        to avoid duplicate column names in the join.
    match_score : float, default=0
        Distance score between the closest matches that will be accepted.
        In a [0, 1] interval. 1 means that only a perfect match will be
        accepted, and zero means that the closest match will be accepted,
        no matter how distant.
        For numerical joins, this defines the maximum Euclidean distance
        between the matches.
    analyzer : {'word', 'char', 'char_wb'}, default=`char_wb`
        Analyzer parameter for the CountVectorizer used for
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
    AggJoiner :
        Aggregate auxiliary dataframes before joining them on a base dataframe.

    fuzzy_join :
        Join two tables (dataframes) based on approximate column matching.

    get_ken_embeddings :
        Download vector embeddings for many common entities (cities,
        places, people...).

    Examples
    --------
    >>> X = pd.DataFrame(['France', 'Germany', 'Italy'], columns=['Country'])
    >>> X
       Country
    0   France
    1  Germany
    2    Italy

    >>> aux_table = pd.DataFrame([['germany', 84_000_000],
    ...                         ['france', 68_000_000],
    ...                         ['italy', 59_000_000]],
    ...                         columns=['Country', 'Population'])
    >>> aux_table
       Country  Population
    0  germany    84000000
    1   france    68000000
    2    italy    59000000

    >>> joiner = Joiner(aux_table, key='Country', suffix='_aux')

    >>> augmented_table = joiner.fit_transform(X)
    >>> augmented_table
       Country Country_aux  Population
    0   France      france    68000000
    1  Germany     germany    84000000
    2    Italy       italy    59000000
    """

    def __init__(
        self,
        aux_table,
        *,
        main_key=None,
        aux_key=None,
        key=None,
        suffix="",
        matching=DEFAULT_MATCHING,
        analyzer="char_wb",
        ngram_range=(2, 4),
    ):
        self.aux_table = aux_table
        self.main_key = main_key
        self.aux_key = aux_key
        self.key = key
        self.suffix = suffix
        self.matching = clone(matching) if matching is DEFAULT_MATCHING else matching
        self.analyzer = analyzer
        self.ngram_range = ngram_range

    def fit(self, X: pd.DataFrame, y=None) -> "Joiner":
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
        Joiner
            Fitted Joiner instance (self).
        """
        self._main_key, self._aux_key = _join_utils.check_key(
            self.main_key, self.aux_key, self.key
        )
        _join_utils.check_missing_columns(X, self._main_key, "'X' (the main table)")
        _join_utils.check_missing_columns(self.aux_table, self._aux_key, "'aux_table'")
        self.vectorizer_ = _make_vectorizer(self.aux_table[self._aux_key])
        aux = self.vectorizer_.fit_transform(self.aux_table[self._aux_key])
        main = self.vectorizer_.transform(
            X[self._main_key].set_axis(self._aux_key, axis="columns")
        )
        self.matching_ = clone(self.matching).fit(aux, main)
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
        _join_utils.check_missing_columns(X, self._main_key, "'X' (the main table)")
        main = self.vectorizer_.transform(
            X[self._main_key].set_axis(self._aux_key, axis="columns")
        )
        match_result = self.matching_.match(main)
        return match_result
