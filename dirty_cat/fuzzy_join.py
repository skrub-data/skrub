# TODO: add HashingVectorizer as an option.
# TODO 2: add suffixes to column names.

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
from sklearn.base import BaseEstimator, TransformerMixin


class FuzzyJoin(BaseEstimator, TransformerMixin):
    """
    Join tables based on categorical string columns as joining keys.

    Parameters
    ----------

    analyzer : str, default='char_wb'.
        Analyzer parameter for the CountVectorizer.
        Options: {‘word’, ‘char’, ‘char_wb’}, describing whether the matrix V
        to factorize should be made of word counts or character n-gram counts.
        Option ‘char_wb’ creates character n-grams only from text inside word
        boundaries; n-grams at the edges of words are padded with space.
    ngram_range : tuple (min_n, max_n), default=(2, 4)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n.
        will be used.

    """

    def __init__(self, analyzer="char_wb", ngram_range=(2, 4)):
        self.ngram_range = ngram_range
        self.analyzer = analyzer

    def join(self, left_table, right_table, on, return_distance=False):
        """Join left and right table.

        Parameters
        ----------
        left_table: pd.DataFrame
            Table on which the join will be performed.
        right_table: pd.DataFrame
            Table that will be joined.
        on: list
            List of left and right table column names on which
            the matching will be perfomed.
        distance: boolean
            Wheter to return distance between nearest matched categories.

        Returns:
        --------
            joined: pd.DataFrame
                A joined table.

        """

        if self.analyzer not in ["char", "word", "char_wb"]:
            raise ValueError(
                "analyzer should be either 'char', 'word' or",
                f"'char_wb', got {self.analyzer}",
            )
        if not isinstance(on, list):
            raise ValueError(
                f"value {on} was specified for parameter 'on', "
                "which has invalid type, expected list of column names."
            )
        if len(on) == 1:
            left_col = on[0]
            right_col = on[0]
        elif len(on) == 2:
            left_col = on[0]
            right_col = on[1]
        else:
            raise ValueError(
                f"List {on} was specified for parameter 'on', "
                "the list has invalid number of elements."
            )
        right_clean = right_table[right_col]
        joined = pd.DataFrame(left_table[left_col], columns=[left_col, "col_to_embed"])

        enc = CountVectorizer(analyzer=self.analyzer, ngram_range=self.ngram_range)
        left_enc = enc.fit_transform(left_table[left_col])
        right_enc = enc.transform(right_table[right_col])
        left_enc = TfidfTransformer().fit_transform(left_enc)
        right_enc = TfidfTransformer().fit_transform(right_enc)

        # Find closest neighbor using KNN :
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(right_enc)
        distance, neighbors = neigh.kneighbors(left_enc, return_distance=True)
        idx_closest = np.ravel(neighbors)

        for idx in left_table.index:
            joined.loc[idx, "col_to_embed"] = right_clean[idx_closest[idx]]
        if return_distance:
            return joined, distance
        else:
            return joined
