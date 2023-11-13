"""
The Joiner provides fuzzy joining as a scikit-learn transformer.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from skrub import _join_utils
from skrub._datetime_encoder import DatetimeEncoder
from skrub._matching import TargetNeighbor

DEFAULT_MATCHING = TargetNeighbor()
DEFAULT_STRING_ENCODER = make_pipeline(
    HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4)), TfidfTransformer()
)
_DATETIME_ENCODER = DatetimeEncoder(resolution=None, add_total_seconds=True)


def _make_vectorizer(table, string_encoder):
    transformers = [
        (clone(string_encoder), c)
        for c in table.select_dtypes(include=["string", "category", "object"]).columns
    ]
    num_columns = table.select_dtypes(include="number").columns
    if not num_columns.empty:
        transformers.append((StandardScaler(), num_columns))
    dt_columns = table.select_dtypes("datetime").columns
    if not dt_columns.empty:
        transformers.append(
            (make_pipeline(clone(_DATETIME_ENCODER), StandardScaler()), dt_columns)
        )
    return make_column_transformer(*transformers)


class Joiner(TransformerMixin, BaseEstimator):
    """Augment a main table by fuzzy joining an auxiliary table to it.

    Given an auxiliary table and matching column names, fuzzy join it to the main
    table.
    The principle is as follows:

    1. The auxiliary table and the matching column names are provided at initialisation.
    2. The main table is provided for fitting, and will be joined
       when ``Joiner.transform`` is called.

    It is advised to use hyperparameter tuning tools such as GridSearchCV
    to determine the best `max_dist` parameter, as this can significantly
    improve results.

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
    max_dist : float, default=1.0
        Maximum acceptable (rescaled) distance between a row in the
        ``main_table`` and its nearest neighbor in the ``aux_table``. Rows that
        are farther apart are not considered to match. By default, the distance
        is rescaled so that a value between 0 and 1 is typically a good choice,
        although rescaled distances can be greater than 1. See the description
        of ``maching`` for details on available rescaling strategies.
        ``None`` or ``"inf"`` is interpreted as ``float("inf")``.
    matching : fuzzy matching and distance rescaling strategy
        TODO this could also be chosen by providing a string?
        TODO expand description.
    string_encoder : scikit-learn transformer for text
        By default a ``HashingVectorizer`` combined with a ``TfidfTransforme``
        is used.

    See Also
    --------
    AggJoiner :
        Aggregate auxiliary dataframes before joining them on a base dataframe.

    fuzzy_join :
        Join two tables (dataframes) based on approximate column matching. This
        is the same functionality as provided by the ``Joiner`` but exposed as
        a function rather than a transformer.

    Examples
    --------
    >>> import pandas as pd
    >>> main_table = pd.DataFrame({"Country": ["France", "Italia", "Spain"]})
    >>> aux_table = pd.DataFrame( {"Country": ["Germany", "France", "Italy"],
    ...                            "Capital": ["Berlin", "Paris", "Rome"]} )
    >>> main_table
      Country
    0  France
    1  Italia
    2   Spain
    >>> aux_table
       Country Capital
    0  Germany  Berlin
    1   France   Paris
    2    Italy    Rome
    >>> joiner = Joiner(aux_table, key="Country", suffix="_capitals")
    >>> joiner.fit_transform(main_table)
      Country Country_capitals Capital_capitals
    0  France           France            Paris
    1  Italia            Italy             Rome
    2   Spain              NaN              NaN
    """

    _match_info_keys = ["distance", "rescaled_distance", "match_accepted"]
    _match_info_key_renaming = {k: f"skrub.Joiner.{k}" for k in _match_info_keys}
    match_info_columns = list(_match_info_key_renaming.values())

    def __init__(
        self,
        aux_table,
        *,
        main_key=None,
        aux_key=None,
        key=None,
        suffix="",
        max_dist=1.0,
        matching=DEFAULT_MATCHING,
        string_encoder=DEFAULT_STRING_ENCODER,
        insert_match_info=False,
    ):
        self.aux_table = aux_table
        self.main_key = main_key
        self.aux_key = aux_key
        self.key = key
        self.suffix = suffix
        self.max_dist = max_dist
        self.matching = clone(matching) if matching is DEFAULT_MATCHING else matching
        self.string_encoder = (
            clone(string_encoder)
            if string_encoder is DEFAULT_STRING_ENCODER
            else string_encoder
        )
        self.insert_match_info = insert_match_info

    def _check_max_dist(self):
        if (
            self.max_dist is None
            or isinstance(self.max_dist, str)
            and self.max_dist == "inf"
        ):
            self._max_dist = np.inf
        else:
            self._max_dist = self.max_dist

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
        del y
        self._main_key, self._aux_key = _join_utils.check_key(
            self.main_key, self.aux_key, self.key
        )
        _join_utils.check_missing_columns(X, self._main_key, "'X' (the main table)")
        _join_utils.check_missing_columns(self.aux_table, self._aux_key, "'aux_table'")
        self._check_max_dist()
        self.vectorizer_ = _make_vectorizer(
            self.aux_table[self._aux_key], self.string_encoder
        )
        # TODO: fast path if max_dist == 0 and not return_matching_info, don't
        # vectorize nor fit matching just do normal equijoin
        aux = self.vectorizer_.fit_transform(self.aux_table[self._aux_key])
        # TODO: add a fit_transform to avoid vectorizing main twice; also maybe
        # skip in fit if the matching does not need it
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
        del y
        _join_utils.check_missing_columns(X, self._main_key, "'X' (the main table)")
        main = self.vectorizer_.transform(
            X[self._main_key].set_axis(self._aux_key, axis="columns")
        )
        match_result = self.matching_.match(main, self._max_dist)
        aux_table = self.aux_table.rename(
            columns={c: f"{c}{self.suffix}" for c in self.aux_table.columns}
        )
        matching_col = match_result["index"].copy()
        matching_col[~match_result["match_accepted"]] = -1
        join = pd.merge(
            X,
            aux_table,
            left_on=matching_col,
            right_index=True,
            suffixes=("", ""),
            how="left",
        )
        if self.insert_match_info:
            # TODO maybe let the matching strategy decide which keys to insert
            # (eg number of competitors in neighborhood, distance to closest
            # competitor etc)
            for info_key, info_col_name in self._match_info_key_renaming.items():
                join[info_col_name] = match_result[info_key]
        return join
