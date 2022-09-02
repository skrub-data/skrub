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


def FuzzyJoin(left_table, right_table, on, return_distance=False, analyzer="char_wb",
              ngram_range=(2, 4), precision='nearest', precision_threshold=0.5, suffixes=('_l', '_r'), keep='all',):
    """
    Join tables based on categorical string columns as joining keys.

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
        Analyzer parameter for the CountVectorizer.
        Options: {`word`, `char`, `char_wb`}, describing whether the matrix V
        to factorize should be made of word counts or character n-gram counts.
        Option `char_wb` creates character n-grams only from text inside word
        boundaries; n-grams at the edges of words are padded with space.
    ngram_range : tuple (min_n, max_n), default=(2, 4)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n.
        will be used.
    precision : {`nearest`, `2dballtree`}, default=`nearest`
        Type of measure that is used to determine the precision of the joined entities.
        If `nearest`, returns the neirest neighbor match. 
        If `2dballtree`, return the nearest neighbor if the estimated precision is
        under the precision threshold.
    precision_threshold : float, default=0.5
        Used only if precision is `2dballtree`. Determines the level of precision
        required to match the two column values. If not matched, all columns have
        `nan`'s.
    suffixes: tuple, default=('_x', '_y')
            A list of strings indicating the suffix to add when overlaping column names.
    keep: {'left', 'right', 'all'}, default='all'
            Wheter to keep the matching columns from the left, right or all tables. 

    Returns:
    --------
    joined: pd.DataFrame
        A joined table.
    distance : 

        """

    lt = left_table.reset_index(drop=True).fillna('').copy()
    rt = right_table.reset_index(drop=True).fillna('').copy()

    if analyzer not in ["char", "word", "char_wb"]:
        raise ValueError(
            "analyzer should be either 'char', 'word' or",
            f"'char_wb', got {analyzer}",
        )

    if precision not in ["nearest", "2dball"]:
        raise ValueError(
            "precision should be either 'nearest' or",
            f"'2dballtree', got {precision}",
        )

    if keep not in ["left", "right", "all"]:
        raise ValueError(
            "keep should be either 'left', 'right' or",
            f"'all', got {keep}",
        )

    if len(suffixes)!=2:
        raise ValueError(f"Number of suffixes specified is different than two: {suffixes}")
    lsuffix, rsuffix = suffixes
    if not lsuffix and not rsuffix:
        raise ValueError(f"Tuple {suffixes} has invalid number of elements.")
    overlap_cols = lt._info_axis.intersection(rt._info_axis)
    if len(overlap_cols)>0:
        for i in range(len(overlap_cols)):
            new_name_l = overlap_cols[i] + lsuffix
            new_name_r = overlap_cols[i] + rsuffix
            lt.rename(columns = {overlap_cols[i]:new_name_l}, inplace = True)
            rt.rename(columns = {overlap_cols[i]:new_name_r}, inplace = True)
            if len(on)==2 and overlap_cols[i] in on[0]:
                on[0] = new_name_l
            if len(on)==2 and overlap_cols[i] in on[1]:
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
            f"List {on} was specified for parameter 'on', "
            "the list has invalid number of elements."
        )

    cols = list(lt.columns) + list(rt.columns)
    joined = pd.DataFrame(lt, columns=cols)

    enc = CountVectorizer(analyzer=analyzer, ngram_range=ngram_range)
    left_enc = enc.fit_transform(lt[left_col])
    right_enc = enc.transform(rt[right_col])
    
    left_enc = TfidfTransformer().fit_transform(left_enc)
    right_enc = TfidfTransformer().fit_transform(right_enc)

    # Find closest neighbor using KNN :
    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(right_enc)
    distance, neighbors = neigh.kneighbors(left_enc, return_distance=True)
    idx_closest = np.ravel(neighbors)

    if precision == 'nearest':
        for idx in lt.index:
            joined.loc[idx, rt.columns] = list(rt.iloc[idx_closest[idx]])

    if precision == '2dball':
        prec = []
        for i in range(left_enc.shape[0]):
            # Find all neighbors in a 2dball radius:
            dist = 2 * distance[i]
            n_neigh = NearestNeighbors(radius=dist)
            n_neigh.fit(right_enc)
            rng = n_neigh.radius_neighbors(left_enc[i])
            # Distances to closest neighbors:
            # twodball_dist = rng[0][0]
            # Their indices:
            twodball_pts = rng[1][0]
            prec.append(1 / len(twodball_pts))
            # Compute the estimated True Positive, False Positive:
            TP = sum(prec)
            FP = sum([1 - value for value in prec])
            # Finally, estimated precision and recall:
            est_precision = TP/(TP+FP)
            # est_recall = 1 - est_precision
        for idx in lt.index:
            if prec[idx] >= precision_threshold:
                joined.loc[idx, rt.columns] = list(rt.iloc[idx_closest[idx]])
            else:
                joined.loc[idx, rt.columns] = np.nan

    joined = joined.replace(r'^\s*$', np.nan, regex=True)
    if keep=='left':
        joined.drop(columns=[right_col], inplace=True)
    if keep=='right':
        joined.drop(columns=[left_col], inplace=True)
    if return_distance:
        return joined, distance
    else:
        return joined
