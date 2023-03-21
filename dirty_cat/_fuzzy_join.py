"""
Fuzzy joining tables using string columns.
The principle is as follows:
  1. We embed and transform the key string columns using
  HashingVectorizer and TfifdTransformer.
  2. For each category, we use the nearest neighbor method to find its closest
  neighbor and establish a match.
  3. We match the tables using the previous information.
Categories from the two tables that share many sub-strings (n-grams)
have greater probability of beeing matched together. The join is based on
morphological similarities between strings.
"""

import warnings
from typing import Literal, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import vstack
from sklearn.feature_extraction.text import (
    HashingVectorizer,
    TfidfTransformer,
    _VectorizerMixin,
)
from sklearn.neighbors import NearestNeighbors


def fuzzy_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    how: Literal["left", "right"] = "left",
    left_on: Union[str, None] = None,
    right_on: Union[str, None] = None,
    on: Union[str, None] = None,
    encoder: Union[Literal["hashing"], _VectorizerMixin] = None,
    analyzer: Literal["word", "char", "char_wb"] = "char_wb",
    ngram_range: Tuple[int, int] = (2, 4),
    return_score: bool = False,
    match_score: float = 0,
    drop_unmatched: bool = False,
    sort: bool = False,
    suffixes: Tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:
    """
    Join two tables categorical string columns based on approximate
    matching and using morphological similarity.

    Parameters
    ----------
    left : :class:`~pandas.DataFrame`
        A table to merge.
    right : :class:`~pandas.DataFrame`
        A table used to merge with.
    how: typing.Literal["left", "right"], default=`left`
        Type of merge to be performed. Note that unlike :func:`~pandas.merge`,
        only "left" and "right" are supported so far, as the fuzzy-join comes
        with its own mechanism to resolve lack of correspondence between
        left and right tables.
    left_on : str, optional, default=None
        Name of left table column to join.
    right_on : str, optional, default=None
        Name of right table key column to join
        with left table key column.
    on : str, optional, default=None
        Name of common left and right table join key columns.
        Must be found in both DataFrames. Use only if `left_on`
        and `right_on` parameters are not specified.
    encoder: Union[Literal["hashing"], _VectorizerMixin], default=None,
        Encoder parameter for the Vectorizer.
        Options: {None, `_VectorizerMixin`}. If None, the
        encoder will use the `HashingVectorizer`. It is possible to pass a
        `_VectorizerMixin` custom object to tweak the parameters of the encoder.
    analyzer : {"word", "char", "char_wb"}, optional, default=`char_wb`
        Analyzer parameter for the HashingVectorizer passed to
        the encoder and used for the string similarities.
        Options: {`word`, `char`, `char_wb`}, describing whether the matrix V
        to factorize should be made of word counts or character n-gram counts.
        Option `char_wb` creates character n-grams only from text inside word
        boundaries; n-grams at the edges of words are padded with space.
    ngram_range : int 2-tuple, optional, default=(2, 4)
        The lower and upper boundary of the range of n-values for different
        n-grams used in the string similarity. All values of n such
        that min_n <= n <= max_n will be used.
    return_score : boolean, default=True
        Whether to return matching score based on the distance between
        the nearest matched categories.
    match_score : float, default=0.0
        Distance score between the closest matches that will be accepted.
        In a [0, 1] interval. 1 means that only a perfect match will be
        accepted, and zero means that the closest match will be accepted,
        no matter how distant.
    drop_unmatched : boolean, default=False
        Remove categories for which a match was not found in the two tables.
    sort : boolean, default=False
        Sort the join keys lexicographically in the result DataFrame.
        If False, the order of the join keys depends on the join type
        (`how` keyword).
    suffixes : str 2-tuple, default=('_x', '_y')
        A list of strings indicating the suffix to add when overlaping
        column names.

    Returns
    -------
    df_joined : :class:`~pandas.DataFrame`
        The joined table returned as a DataFrame. If `return_score` is True,
        another column will be added to the DataFrame containing the
        matching scores.

    See Also
    --------
    :class:`~dirty_cat.FeatureAugmenter` :
        Transformer to enrich a given table via one or more fuzzy joins to
        external resources.

    Notes
    -----
    For regular joins, the output of fuzzy_join is identical
    to :func:`~pandas.merge`, except that both key columns are returned.

    Joining on indexes and multiple columns is not supported.

    When `return_score=True`, the returned :class:`~pandas.DataFrame` gives
    the distances between the closest matches in a [0, 1] interval.
    0 corresponds to no matching n-grams, while 1 is a
    perfect match.

    When we use `match_score=0`, the function will be forced to impute the
    nearest match (of the left table category) across all possible matching
    options in the right table column.

    When the neighbors are distant, we may use the `match_score` parameter
    with a value bigger than 0 to define the minimal level of matching
    score tolerated. If it is not reached, matches will be
    considered as not found and NaN values will be imputed.

    Examples
    --------
    >>> df1 = pd.DataFrame({'a': ['ana', 'lala', 'nana'], 'b': [1, 2, 3]})
    >>> df2 = pd.DataFrame({'a': ['anna', 'lala', 'ana', 'nnana'], 'c': [5, 6, 7, 8]})

    >>> df1
        a  b
    0   ana  1
    1  lala  2
    2  nana  3

    >>> df2
        a    c
    0  anna  5
    1  lala  6
    2  ana   7
    3  nnana 8

    To do a simple join based on the nearest match:

    >>> fuzzy_join(df1, df2, on='a')
        a_x  b   a_y    c
    0   ana  1   ana    7
    1  lala  2  lala    6
    2  nana  3  nnana   8

    When we want to accept only a certain match precision,
    we can use the `match_score` argument:

    >>> fuzzy_join(df1, df2, on='a', match_score=1, return_score=True)
        a_x  b   a_y    c  matching_score
    0   ana  1   ana  7.0  1.000000
    1  lala  2  lala  6.0  1.000000
    2  nana  3   NaN  NaN  0.532717

    As expected, the category "nana" has no exact match (`match_score=1`).
    """

    warnings.warn("This feature is still experimental.")

    if analyzer not in ["char", "word", "char_wb"]:
        raise ValueError(
            f"analyzer should be either 'char', 'word' or 'char_wb', got {analyzer!r}",
        )

    if encoder is not None:
        if not issubclass(encoder.__class__, _VectorizerMixin):
            raise ValueError(f"encoder should be a vectorizer object, got {encoder!r}")

    if how not in ["left", "right"]:
        raise ValueError(
            f"how should be either 'left' or 'right', got {how!r}",
        )

    if not isinstance(match_score, (int, float)):
        raise TypeError("match_score has invalid type, expected integer or float")

    for param in [on, left_on, right_on]:
        if param is not None and not isinstance(param, str):
            raise TypeError(
                "Parameter 'left_on', 'right_on' or 'on' has invalid type, expected"
                " string"
            )

    # TODO: enable joining on multiple keys as in pandas.merge
    if on is not None:
        left_col = on
        right_col = on
    elif left_on is not None and right_on is not None:
        left_col = left_on
        right_col = right_on
    else:
        raise KeyError(
            "Required parameter missing: either parameter"
            "'on' or the pair 'left_on', 'right_on' should be specified."
        )

    if how == "left":
        main_table = left.reset_index(drop=True).copy()
        aux_table = right.reset_index(drop=True).copy()
        main_col = left_col
        aux_col = right_col
    else:
        main_table = right.reset_index(drop=True).copy()
        aux_table = left.reset_index(drop=True).copy()
        main_col = right_col
        aux_col = left_col

    # Make sure that the column types are string and categorical:
    main_col_clean = main_table[main_col].astype(str)
    aux_col_clean = aux_table[aux_col].astype(str)

    # Warn if presence of missing values
    if main_table[main_col].isna().any():
        warnings.warn(
            "You are merging on missing values."
            " The output correspondence will be random or missing."
            " To avoid unexpected errors you can drop them.",
            UserWarning,
        )

    all_cats = pd.concat([main_col_clean, aux_col_clean], axis=0).unique()

    if encoder is None:
        enc = HashingVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    else:
        enc = encoder

    enc_cv = enc.fit(all_cats)
    main_enc = enc_cv.transform(main_col_clean)
    aux_enc = enc_cv.transform(aux_col_clean)

    all_enc = vstack((main_enc, aux_enc))

    tfidf = TfidfTransformer().fit(all_enc)
    main_enc = tfidf.transform(main_enc)
    aux_enc = tfidf.transform(aux_enc)

    # Find nearest neighbor using KNN :
    neigh = NearestNeighbors(n_neighbors=1)

    neigh.fit(aux_enc)
    distance, neighbors = neigh.kneighbors(main_enc, return_distance=True)
    idx_closest = np.ravel(neighbors)

    main_table["fj_idx"] = idx_closest
    aux_table["fj_idx"] = aux_table.index

    norm_distance = 1 - (distance / 2)
    if drop_unmatched:
        main_table = main_table[match_score <= norm_distance]
        norm_distance = norm_distance[match_score <= norm_distance]
    else:
        main_table.loc[np.ravel(match_score > norm_distance), "fj_nan"] = 1

    if sort:
        main_table.sort_values(by=[main_col], inplace=True)

    # To keep order of columns as in pandas.merge (always left table first)
    if how == "left":
        df_joined = pd.merge(
            main_table, aux_table, on="fj_idx", suffixes=suffixes, how=how
        )
    else:
        df_joined = pd.merge(
            aux_table, main_table, on="fj_idx", suffixes=suffixes, how=how
        )

    if drop_unmatched:
        df_joined.drop(columns=["fj_idx"], inplace=True)
    else:
        idx = df_joined.index[df_joined["fj_nan"] == 1]
        if len(idx) != 0:
            df_joined.iloc[idx, df_joined.columns.get_loc("fj_idx") :] = np.NaN
        df_joined.drop(columns=["fj_idx", "fj_nan"], inplace=True)

    if return_score:
        df_joined = pd.concat(
            [df_joined, pd.DataFrame(norm_distance, columns=["matching_score"])], axis=1
        )

    return df_joined
