"""
The Joiner provides fuzzy joining as a scikit-learn transformer.
"""

from functools import partial

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import make_column_transformer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.utils.fixes import parse_version
from sklearn.utils.validation import check_is_fitted

from . import _dataframe as sbd
from . import _join_utils, _matching, _utils
from . import _selectors as s
from ._check_input import CheckInputDataFrame
from ._datetime_encoder import DatetimeEncoder
from ._table_vectorizer import TableVectorizer
from ._to_str import ToStr
from ._wrap_transformer import wrap_transformer

DEFAULT_STRING_ENCODER = make_pipeline(
    FunctionTransformer(partial(sbd.fill_nulls, value="")),
    ToStr(),
    HashingVectorizer(analyzer="char_wb", ngram_range=(2, 4)),
    TfidfTransformer(),
)
_DATETIME_ENCODER = DatetimeEncoder(resolution=None, add_total_seconds=True)


_MATCHERS = {
    "random_pairs": _matching.RandomPairs,
    "second_neighbor": _matching.OtherNeighbor,
    "self_join_neighbor": _matching.SelfJoinNeighbor,
    "no_rescaling": _matching.Matching,
}
DEFAULT_REF_DIST = "random_pairs"


def _compat_df(df):
    # In scikit-learn versions older than 1.4, the ColumnTransformer fails on
    # polars dataframes. Here it is only applied as an internal step on the
    # joining columns, and we get the output as a numpy array or sparse matrix.
    # Therefore on old scikit-learn versions we convert the joining columns to
    # pandas before vectorizing them.
    if parse_version(sklearn.__version__) < parse_version("1.4"):
        return sbd.to_pandas(df)
    return df


def _make_vectorizer(table, string_encoder, rescale):
    """Construct the transformer used to vectorize joining columns.

    The resulting ColumnTransformer applies TFIDF transformation to string
    columns, DatetimeEncoder to datetimes and passthrough to numeric columns.
    In addition if `rescale` is `True`, a StandardScaler is applied to
    numeric and datetime columns.
    """
    skrubber = TableVectorizer(
        datetime="passthrough",
        low_cardinality="passthrough",
        high_cardinality="passthrough",
        numeric="passthrough",
    )
    table = skrubber.fit_transform(table)
    cols = skrubber.kind_to_columns_
    transformers = [
        (clone(string_encoder), c)
        for c in cols["high_cardinality"] + cols["low_cardinality"]
    ]
    if cols["numeric"]:
        transformers.append(
            (StandardScaler() if rescale else "passthrough", cols["numeric"])
        )
    if cols["datetime"]:
        transformers.append(
            (
                make_pipeline(
                    wrap_transformer(_DATETIME_ENCODER, s.all()),
                    StandardScaler() if rescale else "passthrough",
                ),
                cols["datetime"],
            )
        )
    return make_pipeline(skrubber, make_column_transformer(*transformers))


class Joiner(TransformerMixin, BaseEstimator):
    """Augment features in a main table by fuzzy-joining an auxiliary table to it.

    This transformer is initialized with an auxiliary table `aux_table`. It
    transforms a main table by joining it, with approximate ("fuzzy") matching,
    to the auxiliary table. The output of `transform` has the same rows as
    the main table (i.e. as the argument passed to `transform`), but each row
    is augmented with values from the best match in the auxiliary table.

    To identify the best match for each row, values from the matching columns
    (`main_key` and `aux_key`) are vectorized, i.e. represented by vectors of
    continuous values. Then, the Euclidean distances between these vectors are
    computed to find, for each main table row, its nearest neighbor within the
    auxiliary table.

    Optionally, a maximum distance threshold, `max_dist`, can be set. Matches
    between vectors that are separated by a distance (strictly) greater than
    `max_dist` will be rejected. We will consider that main table rows that
    are farther than `max_dist` from their nearest neighbor do not have a
    matching row in the auxiliary table, and the output will contain nulls for
    the entries that would normally have come from the auxiliary table (as in a
    traditional left join).

    To make it easier to set a `max_dist` threshold, the distances are
    rescaled by dividing them by a reference distance, which can be chosen with
    `ref_dist`. The default is `'random_pairs'`. The possible choices are:

    'random_pairs'
        Pairs of rows are sampled randomly from the auxiliary table and their
        distance is computed. The reference distance is the first quartile of
        those distances.

    'second_neighbor'
        The reference distance is the distance to the *second* nearest neighbor
        in the auxiliary table.

    'self_join_neighbor'
        Once the match candidate (i.e. the nearest neigbor from the auxiliary
        table) has been found, we find its nearest neighbor in the auxiliary
        table (excluding itself). The reference distance is the distance that
        separates those 2 auxiliary rows.

    'no_rescaling'
        The reference distance is 1.0, i.e. no rescaling of the distances is
        applied.

    Parameters
    ----------
    aux_table : dataframe
        The auxiliary table, which will be fuzzy-joined to the main table when
        calling `transform`.
    key : str or list of str, default=None
        The column names to use for both `main_key` and `aux_key` when they
        are the same. Provide either `key` or both `main_key` and `aux_key`.
    main_key : str or list of str, default=None
        The column names in the main table on which the join will be performed.
        Can be a string if joining on a single column.
        If `None`, `aux_key` must also be `None` and `key` must be provided.
    aux_key : str or list of str, default=None
        The column names in the auxiliary table on which the join will
        be performed. Can be a string if joining on a single column.
        If `None`, `main_key` must also be `None` and `key` must be provided.
    suffix : str, default=""
        Suffix to append to the `aux_table`'s column names. You can use it
        to avoid duplicate column names in the join.
    max_dist : int, float, `None` or `np.inf`, default=`np.inf`
        Maximum acceptable (rescaled) distance between a row in the
        `main_table` and its nearest neighbor in the `aux_table`. Rows that
        are farther apart are not considered to match. By default, the distance
        is rescaled so that a value between 0 and 1 is typically a good choice,
        although rescaled distances can be greater than 1 for some choices of
        `ref_dist`. `None`, `"inf"`, `float("inf")` or `numpy.inf`
        mean that no matches are rejected.
    ref_dist : reference distance for rescaling, default='random_pairs'
        Options are {"random_pairs", "second_neighbor", "self_join_neighbor",
        "no_rescaling"}. See above for a description of each option. To
        facilitate the choice of `max_dist`, distances between rows in
        `main_table` and their nearest neighbor in `aux_table` will be
        rescaled by this reference distance.
    string_encoder : scikit-learn transformer used to vectorize text columns
        By default a `HashingVectorizer` combined with a `TfidfTransformer`
        is used. Here we use raw TF-IDF features rather than transforming them
        for example with `GapEncoder` or `MinHashEncoder` because it is
        faster, these features are only used to find nearest neighbors and not
        used by downstream estimators, and distances between TF-IDF vectors
        have a somewhat simpler interpretation.
    add_match_info : bool, default=True
        Insert some columns whose names start with `skrub_Joiner` containing
        the distance, rescaled distance and whether the rescaled distance is
        above the threshold. Those values can be helpful for an estimator that
        uses the joined features, or to inspect the result of the join and set
        a `max_dist` threshold.

    Attributes
    ----------
    max_dist_ : the maximum distance for a match to be accepted
        Equal to the parameter `max_dist` except that `"inf"` and `None`
        are mapped to `np.inf` (i.e. accept all matches).

    vectorizer_ : scikit-learn ColumnTransformer
        The fitted transformer used to transform the matching columns into
        numerical vectors.

    See Also
    --------
    AggJoiner :
        Aggregate an auxiliary dataframe before joining it on a base dataframe.

    fuzzy_join :
        Join two tables (dataframes) based on approximate column matching. This
        is the same functionality as provided by the `Joiner` but exposed as
        a function rather than a transformer.

    Examples
    --------
    >>> import pandas as pd
    >>> from skrub import Joiner
    >>> main_table = pd.DataFrame({"Country": ["France", "Italia", "Georgia"]})
    >>> aux_table = pd.DataFrame( {"Country": ["Germany", "France", "Italy"],
    ...                            "Capital": ["Berlin", "Paris", "Rome"]} )
    >>> main_table
      Country
    0  France
    1  Italia
    2   Georgia
    >>> aux_table
       Country Capital
    0  Germany  Berlin
    1   France   Paris
    2    Italy    Rome
    >>> joiner = Joiner(
    ...     aux_table,
    ...     key="Country",
    ...     suffix="_aux",
    ...     max_dist=0.8,
    ...     add_match_info=False,
    ... )
    >>> joiner.fit_transform(main_table)
      Country      Country_aux      Capital_aux
    0  France           France            Paris
    1  Italia            Italy             Rome
    2  Georgia              NaN              NaN
    """

    _match_info_keys = ["distance", "rescaled_distance", "match_accepted"]
    _match_info_key_renaming = {k: f"skrub_Joiner_{k}" for k in _match_info_keys}
    match_info_columns = list(_match_info_key_renaming.values())

    def __init__(
        self,
        aux_table,
        *,
        key=None,
        main_key=None,
        aux_key=None,
        suffix="",
        max_dist=np.inf,
        ref_dist=DEFAULT_REF_DIST,
        string_encoder=DEFAULT_STRING_ENCODER,
        add_match_info=True,
    ):
        self.aux_table = aux_table
        self.key = key
        self.main_key = main_key
        self.aux_key = aux_key
        self.suffix = suffix
        self.max_dist = max_dist
        self.ref_dist = ref_dist
        self.string_encoder = (
            clone(string_encoder)
            if string_encoder is DEFAULT_STRING_ENCODER
            else string_encoder
        )
        self.add_match_info = add_match_info

    def _check_max_dist(self):
        if (
            self.max_dist is None
            or isinstance(self.max_dist, str)
            and self.max_dist == "inf"
        ):
            self.max_dist_ = np.inf
        else:
            self.max_dist_ = self.max_dist

    def _check_ref_dist(self):
        if self.ref_dist not in _MATCHERS:
            raise ValueError(
                f"'ref_dist' should be one of {list(_MATCHERS.keys())}. Got"
                f" {self.ref_dist!r}"
            )
        self._matching = _MATCHERS[self.ref_dist]()

    def fit(self, X, y=None):
        """Fit the instance to the main table.

        Parameters
        ----------
        X : dataframe
            The main table, to be joined to the auxiliary ones.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        Joiner
            Fitted Joiner instance (self).
        """
        del y
        self._aux_table = CheckInputDataFrame().fit_transform(self.aux_table)
        self._main_check_input = CheckInputDataFrame()
        X = self._main_check_input.fit_transform(X)
        self._check_ref_dist()
        self._check_max_dist()
        self._main_key, self._aux_key = _join_utils.check_key(
            self.main_key, self.aux_key, self.key
        )
        _join_utils.check_missing_columns(X, self._main_key, "'X' (the main table)")
        _join_utils.check_missing_columns(self._aux_table, self._aux_key, "'aux_table'")
        _join_utils.check_column_name_duplicates(
            X, self._aux_table, self.suffix, main_table_name="X"
        )
        self._right_cols_renaming = f"{{}}{self.suffix}".format
        self.vectorizer_ = _make_vectorizer(
            s.select(self._aux_table, s.cols(*self._aux_key)),
            self.string_encoder,
            rescale=self.ref_dist != "no_rescaling",
        )
        aux = self.vectorizer_.fit_transform(
            _compat_df(s.select(self._aux_table, s.cols(*self._aux_key)))
        )
        self._matching.fit(aux)
        return self

    def transform(self, X, y=None):
        """Transform `X` using the specified encoding scheme.

        Parameters
        ----------
        X : dataframe
            The main table, to be joined to the auxiliary ones.
        y : None
            Unused, only here for compatibility.

        Returns
        -------
        dataframe
            The final joined table.
        """
        del y
        check_is_fitted(self, "vectorizer_")
        X = self._main_check_input.transform(X)
        _join_utils.check_missing_columns(X, self._main_key, "'X' (the main table)")
        _join_utils.check_column_name_duplicates(
            X, self._aux_table, self.suffix, main_table_name="X"
        )
        main = sbd.set_column_names(s.select(X, s.cols(*self._main_key)), self._aux_key)
        main = self.vectorizer_.transform(_compat_df(main))
        match_result = self._matching.match(main, self.max_dist_)
        matching_col = match_result["index"].copy()
        matching_col[~match_result["match_accepted"]] = -1
        token = _utils.random_string()
        left_key_name = f"skrub_left_key_{token}"
        right_key_name = f"skrub_right_key_{token}"
        left = sbd.with_columns(X, **{left_key_name: matching_col})
        right = sbd.with_columns(
            self._aux_table,
            **{right_key_name: np.arange(sbd.shape(self._aux_table)[0], dtype="int64")},
        )
        join = _join_utils.left_join(
            left,
            right,
            left_on=left_key_name,
            right_on=right_key_name,
            rename_right_cols=self._right_cols_renaming,
        )
        join = s.select(join, ~s.cols(left_key_name))
        if self.add_match_info:
            match_info_dict = {}
            for info_key, info_col_name in self._match_info_key_renaming.items():
                match_info_dict[info_col_name] = match_result[info_key]
            join = sbd.with_columns(join, **match_info_dict)
        return join
