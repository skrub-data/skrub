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
    how: Literal["left", "right", "all"] = "all",
    return_score: bool = False,
    analyzer: Literal["word", "char", "char_wb"] = "char_wb",
    ngram_range: Tuple[int, int] = (2, 4),
    match_score: float = 0,
    drop_unmatched=False,
    suffixes: Tuple[str, str] = ("_l", "_r"),
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
    how : typing.Literal['left', 'right', 'all'], default='all'
        Keep the join key columns from the left, right or
        all tables.
    return_score : boolean, default=True
        Wheter to return matching score based on the distance between
        nearest matched categories.
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
    >>> df2 = pd.DataFrame({'a': ['anna', 'lala', 'ana', 'sana'], 'c': [5, 6, 7, 8]})

    >>> df1
        a  b
    0   ana  1
    1  lala  2
    2  nana  3

    >>> df2
        a  c
    0  anna  5
    1  lala  6
    2   ana  7
    3  sana  8

    To do a simple join based on the nearest match:

    >>> fuzzy_join(df1, df2, on='a')
        a_l  b   a_r    c
    0   ana  1   ana   7
    1  lala  2  lala   6
    2  nana  3  sana   8

    When we want to accept only a certain match precison,
    we can use the `match_score` argument:

    >>> fuzzy_join(df1, df2, on='a', match_score=1, return_score=True)
        a_l  b   a_r    c  matching_score
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

    if how not in ["left", "right", "all"]:
        raise ValueError(
            f"how should be either 'left', 'right' or 'all', got {how!r}",
        )

    if len(suffixes) != 2:
        raise ValueError(f"Invalid number of suffixes: expected 2, got {len(suffixes)}")
    lsuffix, rsuffix = suffixes

    for param in [on, left_on, right_on]:
        if param is not None and not isinstance(param, str):
            raise ValueError(
                "Parameter 'left_on', 'right_on' or 'on' has invalid type, expected"
                " string"
            )

    left_table_clean = left.reset_index(drop=True).copy()
    right_table_clean = right.reset_index(drop=True).copy()

    overlap_cols = left_table_clean._info_axis.intersection(
        right_table_clean._info_axis
    )
    if len(overlap_cols) > 0:
        if suffixes[0] == "" and suffixes[1] == "":
            raise ValueError(f"Columns overlap but no suffix specified: {overlap_cols}")
        for i in range(len(overlap_cols)):
            new_name_l = overlap_cols[i] + lsuffix
            new_name_r = overlap_cols[i] + rsuffix
            left_table_clean.rename(columns={overlap_cols[i]: new_name_l}, inplace=True)
            right_table_clean.rename(
                columns={overlap_cols[i]: new_name_r}, inplace=True
            )
            if left_on is not None and overlap_cols[i] in left_on:
                left_on = new_name_l
            if right_on is not None and overlap_cols[i] in right_on:
                right_on = new_name_r

    if on is not None:
        left_col = on + lsuffix
        right_col = on + rsuffix
    elif left_on is not None and right_on is not None:
        left_col = left_on
        right_col = right_on

    # Drop missing values in key columns
    left_table_clean.dropna(subset=[left_col], inplace=True)
    right_table_clean.dropna(subset=[right_col], inplace=True)

    # Make sure that the column types are string and categorical:
    left_table_clean[left_col] = (
        left_table_clean[left_col].astype(str).astype("category")
    )
    right_table_clean[right_col] = (
        right_table_clean[right_col].astype(str).astype("category")
    )

    enc = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range)

    all_cats = pd.concat(
        [left_table_clean[left_col], right_table_clean[right_col]], axis=0
    )

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

    norm_distance = 1 - (distance / 2)

    left_array = np.array(left_table_clean)
    right_array = np.array(right_table_clean)
    joined = np.append(
        left_array,
        np.array(
            [
                right_array[idx_closest[idr]]
                if norm_distance[idr] >= match_score
                else np.tile(np.nan, (right_array.shape[1],))
                for idr in left_table_clean.index
            ]
        ),
        axis=1,
    )

    cols = list(left_table_clean.columns) + list(right_table_clean.columns)
    df_joined = pd.DataFrame(joined, columns=cols).replace(r"^\s*$", np.nan, regex=True)

    duplicate_names = df_joined.columns.duplicated(keep=False)
    if sum(duplicate_names) > 0:
        warnings.warn("Column names overlaps. Please set appropriate suffixes.")
        idx_to_keep = list(np.where(~duplicate_names)[0])

    if return_score:
        df_joined = pd.concat(
            [df_joined, pd.DataFrame(norm_distance, columns=["matching_score"])], axis=1
        )
    if drop_unmatched:
        df_joined.dropna(subset=[left_col, right_col], inplace=True)
    if how == "left":
        if sum(duplicate_names) > 0:
            idx_to_keep.append(np.where(duplicate_names)[0][0])
            idx_to_keep.sort()
            df_joined = df_joined.iloc[:, idx_to_keep]
        else:
            df_joined.drop(columns=[right_col], inplace=True)
    elif how == "right":
        if sum(duplicate_names) > 0:
            idx_to_keep.append(np.where(duplicate_names)[0][1])
            idx_to_keep.sort()
            df_joined = df_joined.iloc[:, idx_to_keep]
        else:
            df_joined.drop(columns=[left_col], inplace=True)
    return df_joined
