"""
Implements deduplication
"""
from typing import Sequence, Optional, Tuple, List, Union
from numpy.typing import NDArray
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import minmax_scale


def deduplicate(
    data: Sequence[str],
    n_clusters: Optional[int] = None,
    ngram_range: Tuple[int, int] = (2, 4),
    analyzer: str = "char_wb",
    return_corrections: bool = False,
    method: str = "average",
) -> Union[List[str], Tuple[List[str], pd.Series]]:
    """

    :param data: Sequence of words with 
    :type data: Sequence
    :param n_clusters: _description_, defaults to None
    :type n_clusters: Optional[int], optional
    :param ngram_range: _description_, defaults to (2, 4)
    :type ngram_range: Tuple[int, int], optional
    :param analyzer: _description_, defaults to "char_wb"
    :type analyzer: str, optional
    :param return_corrections: _description_, defaults to False
    :type return_corrections: bool, optional
    :param method: _description_, defaults to "average"
    :type method: str, optional
    :return: _description_
    :rtype: _type_
    """
    unique_examples, counts = np.unique(data, return_counts=True)
    ex_series = pd.Series(counts, index=unique_examples)
    scaled_sim = compute_ngram_similarity(
        unique_examples, ngram_range=ngram_range, analyzer=analyzer
    )

    dense_distance = squareform(scaled_sim, checks=False)
    Z = linkage(dense_distance, method=method, optimal_ordering=True)
    if n_clusters is None:
        n_clusters = guess_clusters(Z, scaled_sim)
    clstrs = fcluster(Z, n_clusters, criterion="maxclust")

    pd_spell_correct = create_spelling_correction(ex_series, clstrs)
    unrolled_corrections = pd_spell_correct[data]
    if return_corrections:
        return unrolled_corrections, pd_spell_correct
    else:
        return unrolled_corrections


def guess_clusters(Z: NDArray, scaled_sim: NDArray) -> int:
    """_summary_

    :param Z: _description_
    :type Z: NDArray
    :param scaled_sim: _description_
    :type scaled_sim: NDArray
    :return: _description_
    :rtype: int
    """
    max_clusters = scaled_sim.shape[0]
    n_clusters = np.arange(2, max_clusters)
    silhouette_scores = []
    for n_clust in n_clusters:
        labels = fcluster(Z, n_clust, criterion="maxclust")
        silhouette_avg = silhouette_score(scaled_sim, labels, metric="precomputed")
        silhouette_scores.append(silhouette_avg)
    return n_clusters[np.argmax(silhouette_scores)]


def create_spelling_correction(
    count_series: pd.Series, clusters: Sequence
) -> pd.Series:
    """_summary_

    :param count_series: _description_
    :type count_series: pd.Series
    :param clusters: _description_
    :type clusters: Sequence
    :return: _description_
    :rtype: pd.Series
    """
    original_spelling: List[str] = []
    corrected_spelling: List[str] = []
    for cluster in np.unique(clusters):
        sorted_spellings = (
            count_series.loc[clusters == cluster]
            .sort_values(ascending=False)
            .index.values
        )
        original_spelling.extend(sorted_spellings.tolist())
        corrected_spelling.extend(
            np.repeat(sorted_spellings[0], sorted_spellings.shape[0])
        )
    pd_spell_correct = pd.Series(corrected_spelling, index=original_spelling)
    return pd_spell_correct


def compute_ngram_similarity(
    unique_words: NDArray,
    ngram_range: Tuple[int, int] = (2, 4),
    analyzer: str = "char_wb",
) -> NDArray:
    """_summary_

    :param unique_words: _description_
    :type unique_words: Sequence
    :param ngram_range: _description_, defaults to (2, 4)
    :type ngram_range: Tuple[int, int], optional
    :param analyzer: _description_, defaults to "char_wb"
    :type analyzer: str, optional
    :return: _description_
    :rtype: NDArray
    """
    enc = CountVectorizer(ngram_range=ngram_range, analyzer=analyzer)
    encoded = TfidfTransformer().fit_transform(enc.fit_transform(unique_words))

    scaled_sim = minmax_scale(-encoded.dot(encoded.T).todense())
    return scaled_sim