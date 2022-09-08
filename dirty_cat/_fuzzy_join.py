"""
Fuzzy joining tables using string columns.
The principle is as follows:
  1. We transform the key columns strings to vectors using CountVectorizer.
  2. For each category, we use the nearest neighbor method to find its closest
  neighbor and establish a match.
  3. We match the tables using the previous information.
Categories from the two tables that share many n-grams have greater
probability of beeing matched together. The join is based on
morphological similarities between strings.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors
from typing import List, Literal, Tuple


def fuzzy_join(
    left_table: pd.DataFrame,
    right_table: pd.DataFrame,
    on: List[str],
    return_distance: bool = False,
    analyzer: Literal["word", "char", "char_wb"] = "char_wb",
    ngram_range: Tuple[int, int] = (2, 4),
    match_type: Literal["nearest", "radius"] = "nearest",
    match_threshold: float = 0.5,
    suffixes: Tuple[str, str] = ("_l", "_r"),
    keep: Literal["left", "right", "all"] = "all",
) -> pd.DataFrame:
    """
    Join two tables based on categorical string columns as joining keys,
    and approximate matching via string similarity across the two tables.

    Parameters
    ----------

    left_table: pandas.DataFrame
            Table on which the join will be performed.
    right_table: pandas.DataFrame
            Table that will be joined.
    on: list
            List of left and right table column names on which
            the matching will be perfomed.
    return_distance: boolean, default=True
            Wheter to return distance between nearest matched categories.
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
    match_type : typing.Literal["nearest", "radius"], default=`nearest`
        Type of measure that is used to estimate the precision of the joined
        entities.
        If `nearest`, only returns the neirest neighbor match.
        If `radius`, return the nearest neighbor if the estimated precision
        based on the number of neighbors in the 2 times the neirest neighbor
        distance radius is under the precision_threshold.
    match_threshold : float, default=0.5
        Used only if match_type is `radius`. Determines the level of
        precision required to match the two column values. If not matched,
        all columns have `nan`'s.
    suffixes: typing.Tuple[str, str], default=('_x', '_y')
            A list of strings indicating the suffix to add when overlaping
            column names.
    keep: typing.Literal['left', 'right', 'all'], default='all'
            Wheter to keep the matching columns from the left, right or
            all tables.

    Returns:
    --------
    df_joined: pandas.DataFrame
        The joined table returned as a DataFrame.
    distance: bool, default=True
        Whether or not to return the distances to the closest matching
        neighbor.

    Notes
    -----
    There are two main ways to take into account for the similarity between
    categories.
    When we use match_type='nearest', the function will be forced to find the
    nearest match (of the left_table category) across the possible matching
    options in the right_table column.
    When the neighbors are distant, we may use the match_type='radius' option
    with a match_threshold value to define the minimal level of matching
    precision tolerated. If it is not reached, matches will be
    considered as not found and NaN values will be imputed.
    See example below for an illustration.

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
    >>> fuzzy_join(df1, df2, on=['a'], match_type='nearest')
        a_l  b   a_r    c
    0   ana  1   ana   7
    1  lala  2  lala   6
    2  nana  3  sana   8

    When we do not want to ignore the precison of the match,
    we can use the match_type='radius' argument and give a threshold:
    >>> fuzzy_join(df1, df2, on=['a'], match_type='radius', match_threshold=0.3)
        a_l  b   a_r    c
    0   ana  1   ana  7.0
    1  lala  2  lala  6.0
    2  nana  3   NaN  NaN

    As expected, "nana" has no close match and therefore will not be matched.

    """

    lt = left_table.reset_index(drop=True).fillna("").copy()
    rt = right_table.reset_index(drop=True).fillna("").copy()

    if analyzer not in ["char", "word", "char_wb"]:
        raise ValueError(
            f"analyzer should be either 'char', 'word' or 'char_wb', got {analyzer!r}",
        )

    if match_type not in ["nearest", "radius"]:
        raise ValueError(
            f"match_type should be either 'nearest' or 'radius', got {match_type!r}",
        )

    if keep not in ["left", "right", "all"]:
        raise ValueError(f"keep should be either 'left', 'right' or 'all', got {keep!r}",)

    if len(suffixes) != 2:
        raise ValueError(
            "Invalid number of suffixes: expected 2, got {len(suffixes)}"
        )
    lsuffix, rsuffix = suffixes

    overlap_cols = lt._info_axis.intersection(rt._info_axis)
    if len(overlap_cols) > 0:
        for i in range(len(overlap_cols)):
            new_name_l = overlap_cols[i] + lsuffix
            new_name_r = overlap_cols[i] + rsuffix
            lt.rename(columns={overlap_cols[i]: new_name_l}, inplace=True)
            rt.rename(columns={overlap_cols[i]: new_name_r}, inplace=True)
            # Useful in case on[0]==on[1] and overlapping:
            if len(on) == 2 and overlap_cols[i] in on[0]:
                on[0] = new_name_l
            if len(on) == 2 and overlap_cols[i] in on[1]:
                on[1] = new_name_r

    if not isinstance(on, list):
        raise ValueError(
            f"value {on!r} was specified for parameter 'on', "
            "which has invalid type, expected list of column names."
        )
    if len(on) == 1:
        left_col = on[0] + lsuffix
        right_col = on[0] + rsuffix
    elif len(on) == 2:
        left_col = on[0]
        right_col = on[1]
    else:
        raise ValueError(
            "Expected a list with one or two elements for parameter 'on',"
            f"received {len(on)} ({on})."
        )

    enc = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    left_enc = enc.fit_transform(lt[left_col])
    right_enc = enc.transform(rt[right_col])

    left_enc = TfidfTransformer().fit_transform(left_enc)
    right_enc = TfidfTransformer().fit_transform(right_enc)

    # Find nearest neighbor using KNN :
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(right_enc)
    distance, neighbors = neigh.kneighbors(left_enc, return_distance=True)
    idx_closest = np.ravel(neighbors)

    lt_arr = np.array(lt)
    rt_arr = np.array(rt)
    if match_type == "nearest":
        joined = np.append(
            lt_arr, np.array([rt_arr[idx_closest[idr]] for idr in lt.index]), axis=1
        )

    elif match_type == "radius":
        prec = np.zeros(left_enc.shape[0])
        for i in range(left_enc.shape[0]):
            # Find all neighbors in a given radius:
            dist = 2 * distance[i]
            n_neigh = NearestNeighbors(radius=dist)
            n_neigh.fit(right_enc)
            rng = n_neigh.radius_neighbors(left_enc[i])
            # Indices of nearest neighbors:
            twodball_pts = rng[1][0]
            prec[i] = 1 / len(twodball_pts)
        joined = np.append(
            lt_arr,
            np.array(
                [
                    rt_arr[idx_closest[idr]]
                    if prec[idr] >= match_threshold
                    else np.tile(np.nan, (2,))
                    for idr in lt.index
                ]
            ),
            axis=1,
        )

    cols = list(lt.columns) + list(rt.columns)
    df_joined = pd.DataFrame(joined, columns=cols).replace(r"^\s*$", np.nan, regex=True)

    if keep == "left":
        df_joined.drop(columns=[right_col], inplace=True)
    if keep == "right":
        df_joined.drop(columns=[left_col], inplace=True)

    if return_distance:
        return df_joined, distance
    else:
        return df_joined
