"""
The Joiner provides fuzzy joining as a scikit-learn transformer.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from skrub import _join_utils, _matching
from skrub._datetime_encoder import DatetimeEncoder


def _as_str(column):
    return column.fillna("").astype(str)


DEFAULT_STRING_ENCODER = make_pipeline(
    FunctionTransformer(_as_str),
    HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4)),
    TfidfTransformer(),
)
_DATETIME_ENCODER = DatetimeEncoder(resolution=None, add_total_seconds=True)


_MATCHERS = {
    "aux_quartile": _matching.Percentile,
    "second_neighbor": _matching.QueryNeighbor,
    "self_join_neighbor": _matching.TargetNeighbor,
    "worst_match": _matching.MaxDist,
    "no_rescaling": _matching.Matching,
}
DEFAULT_REF_DIST = "aux_quartile"


def _make_vectorizer(table, string_encoder, rescale):
    """Construct the transformer used to vectorize joining columns.

    The resulting ColumnTransformer applies TFIDF transformation to string
    columns, DatetimeEncoder to datetimes and passthrough to numeric columns.
    In addition if ``rescale`` is ``True``, a StandardScaler is applied to
    numeric and datetime columns.
    """
    transformers = [
        (clone(string_encoder), c)
        for c in table.select_dtypes(include=["string", "category", "object"]).columns
    ]
    num_columns = table.select_dtypes(include="number").columns
    if not num_columns.empty:
        transformers.append(
            (StandardScaler() if rescale else "passthrough", num_columns)
        )
    dt_columns = table.select_dtypes(["datetime", "datetimetz"]).columns
    if not dt_columns.empty:
        transformers.append(
            (
                make_pipeline(
                    clone(_DATETIME_ENCODER),
                    StandardScaler() if rescale else "passthrough",
                ),
                dt_columns,
            )
        )
    return make_column_transformer(*transformers)


class Joiner(TransformerMixin, BaseEstimator):
    """Augment features in a main table by fuzzy-joining an auxiliary table to it.

    This transformer is initialized with an auxiliary table ``aux_table``. It
    transforms a main table by joining it, with approximate ("fuzzy") matching,
    to the auxiliary table. The output of ``transform`` has the same rows as
    the main table (ie as the argument passed to ``transform``), but each row
    is augmented with values from the best match in the auxiliary table.

    To identify the best match for each row, values from the matching columns
    (``main_key`` and ``aux_key``) are vectorized, ie represented by vectors of
    continuous values. Then, the Euclidean distances between these vectors are
    computed to find, for each main table row, its nearest neighbor within the
    auxiliary table.

    Optionally, a maximum distance threshold, ``max_dist``, can be set. Matches
    between vectors that are separated by a distance (strictly) greater than
    ``max_dist`` will be rejected. We will consider that main table rows that
    are farther than ``max_dist`` from their nearest neighbor do not have a
    matching row in the auxiliary table, and the output will contain nulls for
    the entries that would normally have come from the auxiliary table (as in a
    traditional left join).

    To make it easier to set a ``max_dist`` threshold, the distances are
    rescaled by dividing them by a reference distance, which can be chosen with
    ``ref_dist``. The default is ``'aux_quartile'``. The possible choices are:

    'aux_quartile'
        Pairs of rows are sampled randomly from the auxiliary table and their
        distance is computed. The reference distance is the first quartile of
        those distances.

    'second_neighbor'
        The reference distance is the distance to the *second* nearest neighbor
        in the auxiliary table.

    'self_join_neighbor'
        Once the match candidate (ie the nearest neigbor from the auxiliary
        table) has been found, we find its nearest neighbor in the auxiliary
        table (excluding itself). The reference distance is the distance that
        separates those 2 auxiliary rows.

    'worst_match'
        The reference distance is the maximum, over all rows in the main table
        provided to ``fit`` (ie ``X`` in ``fit(X)``), of the distance between
        each row and its nearest neighbor in the auxiliary table (ie its
        candidate match). This means that the worst match will have a rescaled
        distance of 1.0.

    'no_rescaling'
        The reference distance is 1.0, ie no rescaling of the distances is
        applied.

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
    max_dist : float, default=np.inf
        Maximum acceptable (rescaled) distance between a row in the
        ``main_table`` and its nearest neighbor in the ``aux_table``. Rows that
        are farther apart are not considered to match. By default, the distance
        is rescaled so that a value between 0 and 1 is typically a good choice,
        although rescaled distances can be greater than 1 for some choices of
        ``ref_dist``. ``None``, ``"inf"``, ``float("inf")`` or ``numpy.inf``
        mean that no matches are rejected.
    ref_dist : reference distance for rescaling, default = 'aux_quartile'
        Options are {"aux_quartile", "second_neighbor", "self_join_neighbor",
        "worst_match", "no_rescaling"}. See above for a description of each
        option. To facilitate the choice of ``max_dist``, distances between
        rows in ``main_table`` and their nearest neighbor in ``aux_table`` will
        be rescaled by this reference distance.
    string_encoder : scikit-learn transformer used to vectorize text columns
        By default a ``HashingVectorizer`` combined with a ``TfidfTransformer``
        is used. Here we use raw TF-IDF features rather than transforming them
        for example with ``GapEncoder`` or ``MinHashEncoder`` because it is
        faster, these features are only used to find nearest neighbors and not
        used by downstream estimators, and distances between TF-IDF vectors
        have a somewhat simpler interpretation.
    insert_match_info : bool, default=True
        Insert some columns whose names start with `skrub_Joiner` containing
        the distance, rescaled distance and whether the rescaled distance is
        above the threshold. Those values can be helpful for an estimator that
        uses the joined features, or to inspect the result of the join and set
        a ``max_dist`` threshold.

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
    >>> joiner = Joiner(
    ...     aux_table,
    ...     key="Country",
    ...     suffix="_capitals",
    ...     max_dist=0.9,
    ...     insert_match_info=False,
    ... )
    >>> joiner.fit_transform(main_table)
      Country Country_capitals Capital_capitals
    0  France           France            Paris
    1  Italia            Italy             Rome
    2   Spain              NaN              NaN
    """

    _match_info_keys = ["distance", "rescaled_distance", "match_accepted"]
    _match_info_key_renaming = {k: f"skrub_Joiner_{k}" for k in _match_info_keys}
    match_info_columns = list(_match_info_key_renaming.values())

    def __init__(
        self,
        aux_table,
        *,
        main_key=None,
        aux_key=None,
        key=None,
        suffix="",
        max_dist=np.inf,
        ref_dist=DEFAULT_REF_DIST,
        string_encoder=DEFAULT_STRING_ENCODER,
        insert_match_info=True,
    ):
        self.aux_table = aux_table
        self.main_key = main_key
        self.aux_key = aux_key
        self.key = key
        self.suffix = suffix
        self.max_dist = max_dist
        self.ref_dist = ref_dist
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

    def _check_ref_dist(self):
        self.matching_ = _MATCHERS[self.ref_dist]()

    def fit(self, X: pd.DataFrame, y=None) -> "Joiner":
        """Fit the instance to the main table.

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
        if self.ref_dist not in _MATCHERS:
            raise ValueError(
                f"ref_dist should be one of {list(_MATCHERS.keys())}, got"
                f" {self.ref_dist!r:}"
            )
        self._main_key, self._aux_key = _join_utils.check_key(
            self.main_key, self.aux_key, self.key
        )
        _join_utils.check_missing_columns(X, self._main_key, "'X' (the main table)")
        _join_utils.check_missing_columns(self.aux_table, self._aux_key, "'aux_table'")
        self._check_max_dist()
        self.vectorizer_ = _make_vectorizer(
            self.aux_table[self._aux_key],
            self.string_encoder,
            rescale=self.ref_dist != "no_rescaling",
        )
        aux = self.vectorizer_.fit_transform(self.aux_table[self._aux_key])
        main = self.vectorizer_.transform(
            X[self._main_key].set_axis(self._aux_key, axis="columns")
        )
        self._check_ref_dist()
        self.matching_.fit(aux, main)
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
            for info_key, info_col_name in self._match_info_key_renaming.items():
                join[info_col_name] = match_result[info_key]
        return join
