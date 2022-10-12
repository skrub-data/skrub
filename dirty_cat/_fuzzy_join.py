"""
Fuzzy joining tables using string columns.
The principle is as follows:
  1. We embed and transform the key string columns using CountVectorizer
  and TfifdTransformer.
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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors


def fuzzy_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    left_on: Union[str, None] = None,
    right_on: Union[str, None] = None,
    on: Union[str, None] = None,
    analyzer: Literal["word", "char", "char_wb"] = "char_wb",
    ngram_range: Tuple[int, int] = (2, 4),
    return_score: bool = False,
    match_score: float = 0,
    drop_unmatched: bool = False,
    suffixes: Tuple[str, str] = ("_x", "_y"),
) -> pd.DataFrame:
    """
    Join two tables categorical string columns based on approximate
    matching and using morphological similarity.

    Parameters
    ----------

    left : pandas.DataFrame
        A table to merge.
    right : pandas.DataFrame
        A table used to merge with.
    left_on : typing.Union[str, None]
        Name of left table column to join.
    right_on : typing.Union[str, None]
        Name of right table key column to join
        with left table key column.
    on : typing.Union[str, None]
        Name of common left and right table join key columns.
        Must be found in both DataFrames. Use only if `left_on`
        and `right_on` parameters are not specified.
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
    return_score : boolean, default=True
        Wheter to return matching score based on the distance between
        nearest matched categories.
    match_score : float, default=0
        Distance score between the closest matches that will be accepted.
        In a [0, 1] interval. Closer to 1 means the matches need to be very
        close to be accepted, and closer to 0 that a bigger matching distance
        is tolerated.
    drop_unmatched : boolean, default=False
        Remove categories for which a match was not found in the two tables.
    suffixes : typing.Tuple[str, str], default=('_x', '_y')
        A list of strings indicating the suffix to add when overlaping
        column names.

    Returns:
    --------
    df_joined: pandas.DataFrame
        The joined table returned as a DataFrame. If `return_score` is True,
        another column will be added to the DataFrame containing the
        matching scores.

    Notes
    -----
    When return_score=True, the returned DataFrame gives
    the distances between closest matches in a [0, 1] interval.
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

    When we want to accept only a certain match precison,
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

    for param in [on, left_on, right_on]:
        if param is not None and not isinstance(param, str):
            raise ValueError(
                "Parameter 'left_on', 'right_on' or 'on' has invalid type, expected"
                " string"
            )

    left_table_clean = left.reset_index(drop=True).copy()
    right_table_clean = right.reset_index(drop=True).copy()

    if on is not None:
        left_col = on
        right_col = on
    elif left_on is not None and right_on is not None:
        left_col = left_on
        right_col = right_on
    else:
        raise ValueError("Parameter 'left_on', 'right_on' or 'on' is missing")

    # Drop missing values in key columns
    left_table_clean.dropna(subset=[left_col], inplace=True)
    right_table_clean.dropna(subset=[right_col], inplace=True)

    # Make sure that the column types are string and categorical:
    left_col_clean = left_table_clean[left_col].astype(str).astype("category")
    right_col_clean = right_table_clean[right_col].astype(str).astype("category")

    enc = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range)

    all_cats = pd.concat([left_col_clean, right_col_clean], axis=0)

    enc_cv = enc.fit(all_cats)
    left_enc = enc_cv.transform(left_table_clean[left_col])
    right_enc = enc_cv.transform(right_table_clean[right_col])

    all_enc = vstack((left_enc, right_enc))

    tfidf = TfidfTransformer().fit(all_enc)
    left_enc = tfidf.transform(left_enc)
    right_enc = tfidf.transform(right_enc)

    # Find nearest neighbor using KNN :
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(right_enc)
    distance, neighbors = neigh.kneighbors(left_enc, return_distance=True)
    idx_closest = np.ravel(neighbors)

    left_table_clean["fj_idx"] = idx_closest
    right_table_clean["fj_idx"] = right_table_clean.index

    norm_distance = np.round(1 - (distance / 2), 6)
    if drop_unmatched:
        left_table_clean = left_table_clean[match_score <= norm_distance]
        norm_distance = norm_distance[match_score <= norm_distance]
    else:
        left_table_clean.loc[np.ravel(match_score > norm_distance), "fj_nan"] = 1

    df_joined = pd.merge(
        left_table_clean, right_table_clean, on="fj_idx", suffixes=suffixes, how="left"
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
