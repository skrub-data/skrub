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


def fuzzy_join(left_table: pd.DataFrame,
               right_table: pd.DataFrame,
               on: List[str],
               return_distance: bool = False,
               analyzer: Literal["word", "char", "char_wb"] = "char_wb",
               ngram_range: Tuple[int, int] = (2, 4),
               precision: Literal["nearest", "radius"] = 'nearest',
               precision_threshold: float = 0.5,
               suffixes: Tuple[str, str] = ('_l', '_r'),
               keep: str = 'all',
               ) -> pd.DataFrame:
    """
    Join two tables based on categorical string columns as joining keys,
    and approximate matching via string similarity across the two tables.

    Parameters
    ----------

    left_table: pd.DataFrame
            Table on which the join will be performed.
    right_table: pd.DataFrame
            Table that will be joined.
    on: list
            List of left and right table column names on which
            the matching will be perfomed.
    return_distance: boolean, default=True
            Wheter to return distance between nearest matched categories.
    analyzer : str, default=`char_wb`
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
    precision : {`nearest`, `radius`}, default=`nearest`
        Type of measure that is used to determine the precision of the joined
        entities.
        If `nearest`, returns the neirest neighbor match.
        If `radius`, return the nearest neighbor if the estimated precision
        based on the number of neighbors in the 2 times the neirest neighbor
        distance radius is under the precision_threshold.
    precision_threshold : float, default=0.5
        Used only if precision is `radius`. Determines the level of
        precision required to match the two column values. If not matched,
        all columns have `nan`'s.
    suffixes: tuple, default=('_x', '_y')
            A list of strings indicating the suffix to add when overlaping
            column names.
    keep: {'left', 'right', 'all'}, default='all'
            Wheter to keep the matching columns from the left, right or
            all tables.

    Returns:
    --------
    joined: pd.DataFrame
        A joined table.
    distance: bool, default=True
        Whether or not to return the distances to the closest matching
        neighbor.

    Notes
    -----
    There are two main ways to take into account for the similarity between
    categories.
    When we use precision='nearest', the function will be forced to find the
    nearest match across the possible options.
    When the neighbors are distant, we may use the precision='radius' option
    with the precision_threshold value to define the minimal level of precision
    every match should have. If this precision is not reached, matches will be
    considered as inexistant and NaN values will be imputed.
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
    >>> fuzzy_join(df1, df2, on=['a'], precision='nearest')
        a_l  b   a_r    c
    0   ana  1   ana   7
    1  lala  2  lala   6
    2  nana  3  sana   8

    When we do not want to ignore the precison of the match,
    we can use the precision='radius' argument and give a threshold:
    >>> fuzzy_join(df1, df2, on=['a'], precision='radius', precision_threshold=0.3)
        a_l  b   a_r    c
    0   ana  1   ana  7.0
    1  lala  2  lala  6.0
    2  nana  3   NaN  NaN

    As expected, "nana" has no close match and therefore will not be matched.

    """

    lt = left_table.reset_index(drop=True).fillna('').copy()
    rt = right_table.reset_index(drop=True).fillna('').copy()

    if analyzer not in ["char", "word", "char_wb"]:
        raise ValueError(
            f"analyzer should be either 'char', 'word' or 'char_wb', got {analyzer}",
        )

    if precision not in ["nearest", "radius"]:
        raise ValueError(
            f"precision should be either 'nearest' or 'radius', got {precision}",
        )

    if keep not in ["left", "right", "all"]:
        raise ValueError(
            f"keep should be either 'left', 'right' or 'all', got {keep}",
        )

    if len(suffixes) != 2:
        raise ValueError("Invalid number of suffixes: expected 2,"
                         f" got {len(suffixes)}")
    lsuffix, rsuffix = suffixes
    if not lsuffix and not rsuffix:
        raise ValueError(f"Suffixes ({suffixes}) has invalid number of elements.")
    overlap_cols = lt._info_axis.intersection(rt._info_axis)
    if len(overlap_cols) > 0:
        for i in range(len(overlap_cols)):
            new_name_l = overlap_cols[i] + lsuffix
            new_name_r = overlap_cols[i] + rsuffix
            lt.rename(columns={overlap_cols[i]: new_name_l}, inplace=True)
            rt.rename(columns={overlap_cols[i]: new_name_r}, inplace=True)
            if len(on) == 2 and overlap_cols[i] in on[0]:
                on[0] = new_name_l
            if len(on) == 2 and overlap_cols[i] in on[1]:
                on[1] = new_name_r

    if not isinstance(on, list):
        raise ValueError(
            f"value {on} was specified for parameter 'on', "
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

    cols = list(lt.columns) + list(rt.columns)
    joined = pd.DataFrame(lt, columns=cols)

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

    if precision == 'nearest':
        for idx in lt.index:
            joined.loc[idx, rt.columns] = list(rt.iloc[idx_closest[idx]])

    if precision == 'radius':
        prec = []
        for i in range(left_enc.shape[0]):
            # Find all neighbors in a given radius:
            dist = 2 * distance[i]
            n_neigh = NearestNeighbors(radius=dist)
            n_neigh.fit(right_enc)
            rng = n_neigh.radius_neighbors(left_enc[i])
            # Indices of nearest neighbors:
            twodball_pts = rng[1][0]
            prec.append(1 / len(twodball_pts))
        for idx in lt.index:
            if prec[idx] >= precision_threshold:
                joined.loc[idx, rt.columns] = list(rt.iloc[idx_closest[idx]])
            else:
                joined.loc[idx, rt.columns] = np.nan

    joined = joined.replace(r'^\s*$', np.nan, regex=True)
    if keep == 'left':
        joined.drop(columns=[right_col], inplace=True)
    if keep == 'right':
        joined.drop(columns=[left_col], inplace=True)
    if return_distance:
        return joined, distance
    else:
        return joined
